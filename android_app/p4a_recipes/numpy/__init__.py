from pythonforandroid.recipe import Recipe, MesonRecipe
from os.path import join
import shutil

NUMPY_NDK_MESSAGE = (
    "In order to build numpy, you must set minimum ndk api (minapi) to `24`.\n"
)

class NumpyRecipe(MesonRecipe):
    version = "v1.26.4"  # ← 改这里：用 1.26.4 替代 2.3.0
    url = "git+https://github.com/numpy/numpy"
    extra_build_args = ["-Csetup-args=-Dblas=none", "-Csetup-args=-Dlapack=none"]
    opt_depends = ["libopenblas"]
    need_stl_shared = True
    min_ndk_api_support = 24

    def get_include(self, arch):
        return join(
            self.ctx.get_python_install_dir(arch.arch), "numpy/core/include",
        )

    def get_recipe_meson_options(self, arch):
        options = super().get_recipe_meson_options(arch)
        options["properties"]["longdouble_format"] = (
            "IEEE_DOUBLE_LE" if arch.arch in ["armeabi-v7a", "x86"] else "IEEE_QUAD_LE"
        )
        return options

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)
        env["_PYTHON_HOST_PLATFORM"] = arch.command_prefix
        env["NPY_DISABLE_SVML"] = "1"
        # 显式设置 numpy include 路径，让 opencv 等依赖能找到头文件
        env["NUMPY_INCLUDE_DIR"] = self.get_include(arch)
        env["CFLAGS"] += f" -I{self.get_include(arch)}"
        env["CXXFLAGS"] += f" -I{self.get_include(arch)}"
        env["TARGET_PYTHON_EXE"] = join(
            Recipe.get_recipe("python3", self.ctx).get_build_dir(arch.arch),
            "android-build",
            "python",
        )
        blas_dir = join(Recipe.get_recipe("libopenblas", self.ctx
                         ).get_build_dir(arch.arch), "build")
        blas_incdir = blas_dir
        blas_libdir = join(blas_dir, "lib")
        env["CXXFLAGS"] += f" -I{blas_incdir} -L{blas_libdir}"

        if 'libopenblas' in self.ctx.recipe_build_order:
            self.extra_build_args = [
                "-Csetup-args=-Dblas=auto",
                "-Csetup-args=-Dlapack=auto",
                "-Csetup-args=-Dallow-noblas=False",
            ]

        return env

    def get_hostrecipe_env(self, arch=None):
        env = super().get_hostrecipe_env(arch=arch)
        env["RANLIB"] = shutil.which("ranlib")
        return env

recipe = NumpyRecipe()
