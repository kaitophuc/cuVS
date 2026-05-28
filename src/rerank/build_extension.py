import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parents[2]
    rerank_dir = root_dir / "src" / "rerank"
    python_bin = Path(sys.executable).resolve()
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", python_bin.parents[1])).resolve()
    nvcc = Path(os.environ.get("NVCC", conda_prefix / "bin" / "nvcc")).resolve()

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    pybind11_includes = subprocess.check_output(
        [str(python_bin), "-m", "pybind11", "--includes"],
        text=True,
    ).split()

    output_dir = rerank_dir / "extensions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"ivfpq_gpu_rerank{ext_suffix}"
    source_path = rerank_dir / "cuda" / "ivfpq_gpu_rerank.cu"

    command = [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "-shared",
        "-Xcompiler=-fPIC",
        *pybind11_includes,
        f"-I{conda_prefix / 'include'}",
        str(source_path),
        "-o",
        str(output_path),
        f"-L{conda_prefix / 'lib'}",
        "-lcudart",
        "-lcuvs",
        "-lraft",
        "-lrmm",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(conda_prefix / "lib"),
    ]

    subprocess.run(command, check=True)
    print(f"Built {output_path}")


if __name__ == "__main__":
    main()
