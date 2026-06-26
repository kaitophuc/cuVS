#include "cagra_data.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

template <typename T>
T arrow_value_or_throw(arrow::Result<T> result, const std::string& where)
{
  if (!result.ok()) {
    throw std::runtime_error(where + ": " + result.status().ToString());
  }
  return std::move(result).ValueOrDie();
}

struct NpyHeader {
  std::string descr;
  bool fortran_order = false;
  std::vector<int64_t> shape;
};

std::string trim(const std::string& s)
{
  const auto begin = s.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(begin, end - begin + 1);
}

NpyHeader parse_npy_header_text(const std::string& header)
{
  NpyHeader out;

  auto find_quoted_value = [&](const std::string& key) {
    const auto key_pos = header.find(key);
    if (key_pos == std::string::npos) {
      throw std::runtime_error("npy header missing key: " + key);
    }
    const auto colon = header.find(':', key_pos);
    const auto quote1 = header.find_first_of("'\"", colon);
    const auto quote2 = header.find_first_of("'\"", quote1 + 1);
    return header.substr(quote1 + 1, quote2 - quote1 - 1);
  };

  out.descr = find_quoted_value("descr");

  const auto fortran_pos = header.find("fortran_order");
  if (fortran_pos == std::string::npos) {
    throw std::runtime_error("npy header missing fortran_order");
  }
  const auto colon = header.find(':', fortran_pos);
  const auto comma = header.find(',', colon);
  const auto value = trim(header.substr(colon + 1, comma - colon - 1));
  out.fortran_order = (value == "True");

  const auto shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("npy header missing shape");
  }
  const auto open = header.find('(', shape_pos);
  const auto close = header.find(')', open);
  std::string shape_text = header.substr(open + 1, close - open - 1);

  std::stringstream ss(shape_text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (!token.empty()) out.shape.push_back(std::stoll(token));
  }

  return out;
}

NpyHeader read_npy_header(std::ifstream& in)
{
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file");
  }

  uint8_t major = 0;
  uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t h16 = 0;
    in.read(reinterpret_cast<char*>(&h16), 2);
    header_len = h16;
  } else if (major == 2 || major == 3) {
    uint32_t h32 = 0;
    in.read(reinterpret_cast<char*>(&h32), 4);
    header_len = h32;
  } else {
    throw std::runtime_error("unsupported npy version");
  }

  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) throw std::runtime_error("failed to read npy header");

  return parse_npy_header_text(header);
}

template <typename T>
std::vector<T> read_npy(const fs::path& path, const std::string& expected_descr, std::vector<int64_t>* shape_out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open " + path.string());

  NpyHeader header = read_npy_header(in);

  if (header.descr != expected_descr) {
    throw std::runtime_error(
      "unexpected dtype in " + path.string() + ": got " + header.descr +
      ", expected " + expected_descr);
  }
  if (header.fortran_order) {
    throw std::runtime_error("Fortran-order npy not supported: " + path.string());
  }

  int64_t count = 1;
  for (int64_t dim : header.shape) count *= dim;

  std::vector<T> data(static_cast<size_t>(count));
  in.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
  if (!in) throw std::runtime_error("failed to read npy payload: " + path.string());

  if (shape_out != nullptr) *shape_out = header.shape;
  return data;
}

fs::path project_root()
{
  return fs::current_path();
}

fs::path data_cache_root()
{
  return fs::path(getenv_or("CUVS_BENCH_DATA_CACHE_DIR", (project_root() / "data_cache").string()));
}

fs::path default_data_cache_dir()
{
  fs::path root = data_cache_root();

  if (fs::exists(root / "dataset.npy") &&
      fs::exists(root / "dataset_ids.npy") &&
      fs::exists(root / "queries.npy")) {
    return root;
  }

  fs::path best;
  fs::file_time_type best_time{};
  bool found = false;

  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;

    const fs::path dir = entry.path();
    if (dir.filename().string().find("default_data_") != 0) continue;
    if (!fs::exists(dir / "dataset.npy")) continue;
    if (!fs::exists(dir / "dataset_ids.npy")) continue;
    if (!fs::exists(dir / "queries.npy")) continue;

    auto t = fs::last_write_time(dir);
    if (!found || t > best_time) {
      best = dir;
      best_time = t;
      found = true;
    }
  }

  if (!found) {
    throw std::runtime_error(
      "Could not find data cache. Run the Python loader once or set "
      "CUVS_BENCH_DATA_CACHE_DIR to a directory containing dataset.npy, "
      "dataset_ids.npy, and queries.npy.");
  }

  return best;
}

LoadedData load_default_data_from_npy_cache()
{
  fs::path dir = default_data_cache_dir();

  std::cout << "Found existing data cache. Loaded: " << dir << "\n";

  std::vector<int64_t> dataset_ids_shape;
  std::vector<int64_t> dataset_shape;
  std::vector<int64_t> queries_shape;

  LoadedData out;
  out.dataset_ids = read_npy<int64_t>(dir / "dataset_ids.npy", "<i8", &dataset_ids_shape);
  out.dataset = read_npy<float>(dir / "dataset.npy", "<f4", &dataset_shape);
  out.queries = read_npy<float>(dir / "queries.npy", "<f4", &queries_shape);

  if (dataset_shape.size() != 2 || queries_shape.size() != 2) {
    throw std::runtime_error("dataset.npy and queries.npy must be 2D");
  }
  if (dataset_shape[1] != VECTOR_DIM || queries_shape[1] != VECTOR_DIM) {
    throw std::runtime_error("unexpected embedding dimension");
  }

  out.n_rows = dataset_shape[0];
  out.dim = dataset_shape[1];
  out.n_queries = queries_shape[0];

  std::cout << "Dataset IDs shape: (" << out.dataset_ids.size() << ")\n";
  std::cout << "Dataset shape: (" << out.n_rows << ", " << out.dim << ")\n";
  std::cout << "Dataset dtype: float32\n";
  std::cout << "Queries shape: (" << out.n_queries << ", " << out.dim << ")\n";
  std::cout << "Queries dtype: float32\n";
  std::cout << "First dataset id: " << out.dataset_ids.front() << "\n";
  std::cout << "Last dataset id: " << out.dataset_ids.back() << "\n";
  std::cout << "Embedding dimension: " << out.dim << "\n";

  return out;
}

fs::path precomputed_neighbors_path()
{
  const fs::path data_dir = fs::path(getenv_or("CUVS_BENCH_DATA_DIR", (project_root() / "openai_large_5m").string()));
  return data_dir / "neighbors.parquet";
}

GroundTruth load_precomputed_ground_truth_from_parquet(int top_k, int query_limit)
{
  const fs::path path = precomputed_neighbors_path();
  if (!fs::exists(path)) {
    throw std::runtime_error("precomputed neighbors parquet does not exist: " + path.string());
  }

  std::shared_ptr<arrow::io::ReadableFile> infile =
    arrow_value_or_throw(arrow::io::ReadableFile::Open(path.string()), "ReadableFile::Open");

  std::unique_ptr<parquet::arrow::FileReader> reader =
    arrow_value_or_throw(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool()),
      "parquet::arrow::OpenFile");

  std::shared_ptr<arrow::Table> table =
    arrow_value_or_throw(reader->ReadTable(), "ReadTable");
  table = arrow_value_or_throw(table->CombineChunks(arrow::default_memory_pool()),
                               "CombineChunks(table)");

  auto id_col = table->GetColumnByName("id");
  auto neighbors_col = table->GetColumnByName("neighbors_id");
  if (!id_col || !neighbors_col) {
    throw std::runtime_error("neighbors parquet missing id or neighbors_id column");
  }
  if (table->num_rows() < query_limit) {
    throw std::runtime_error("query_limit exceeds precomputed neighbor rows");
  }

  if (id_col->num_chunks() != 1 || neighbors_col->num_chunks() != 1) {
    throw std::runtime_error("expected one chunk after CombineChunks(table)");
  }

  auto id_array_base = id_col->chunk(0);
  auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(id_array_base);
  if (!ids) throw std::runtime_error("id column is not int64");

  for (int64_t i = 0; i < query_limit; ++i) {
    if (ids->Value(i) != i) {
      throw std::runtime_error("precomputed query ids do not match first query_limit queries");
    }
  }

  auto neighbors_array_base = neighbors_col->chunk(0);

  GroundTruth gt;
  gt.rows = query_limit;
  gt.cols = top_k;
  gt.neighbors.resize(static_cast<size_t>(query_limit * top_k));

  if (neighbors_array_base->type_id() == arrow::Type::LIST) {
    auto lists = std::dynamic_pointer_cast<arrow::ListArray>(neighbors_array_base);
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(lists->values());
    if (!values) throw std::runtime_error("neighbors_id list values are not int64");

    for (int64_t q = 0; q < query_limit; ++q) {
      const int64_t offset = lists->value_offset(q);
      const int64_t len = lists->value_length(q);
      if (len < top_k) throw std::runtime_error("neighbors_id row has fewer than top_k values");

      for (int j = 0; j < top_k; ++j) {
        gt.neighbors[static_cast<size_t>(q * top_k + j)] = values->Value(offset + j);
      }
    }
  } else if (neighbors_array_base->type_id() == arrow::Type::LARGE_LIST) {
    auto lists = std::dynamic_pointer_cast<arrow::LargeListArray>(neighbors_array_base);
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(lists->values());
    if (!values) throw std::runtime_error("neighbors_id large-list values are not int64");

    for (int64_t q = 0; q < query_limit; ++q) {
      const int64_t offset = lists->value_offset(q);
      const int64_t len = lists->value_length(q);
      if (len < top_k) throw std::runtime_error("neighbors_id row has fewer than top_k values");

      for (int j = 0; j < top_k; ++j) {
        gt.neighbors[static_cast<size_t>(q * top_k + j)] = values->Value(offset + j);
      }
    }
  } else {
    throw std::runtime_error("neighbors_id column is not a list/large_list");
  }

  std::cout << "Loaded precomputed full-dataset ground truth: " << path << "\n";
  return gt;
}

GroundTruth load_required_ground_truth()
{
  return load_precomputed_ground_truth_from_parquet(GROUND_TRUTH_TOP_K, QUERY_LIMIT);
}
