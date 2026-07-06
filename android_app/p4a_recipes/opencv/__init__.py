"""
本地 opencv 配方 — 唯一改动：numpy include 路径从 _core 改为 core
"""
from multiprocessing import cpu_count
from os.path import join
import sh
from pythonforandroid.logger import shprint
from pythonforandroid.recipe import NDKRecipe
from pythonforandroid.util import current_directory, ensure_dir


class OpenCVRecipe(NDKRecipe):
    version = '4.12.0'
    url = 'https://github.com/opencv/opencv/archive/{version}.zip'
    depends = ['numpy']
    # patches 从原配方继承（p4a_build.patch 用于修复 CMake 构建）
    generated_libraries = [
        'libopencv_features2d.so', 'libopencv_imgproc.so', 'libopencv_stitching.so',
        'libopencv_calib3d.so', 'libopencv_flann.so', 'libopencv_ml.so',
        'libopencv_videoio.so', 'libopencv_core.so', 'libopencv_highgui.so',
        'libopencv_objdetect.so', 'libopencv_video.so', 'libopencv_dnn.so',
        'libopencv_imgcodecs.so', 'libopencv_photo.so',
    ]

    def get_lib_dir(self, arch):
        return join(self.get_build_dir(arch.arch), 'build', 'lib', arch.arch)

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['ANDROID_NDK'] = self.ctx.ndk_dir
        env['ANDROID_SDK'] = self.ctx.sdk_dir
        return env

    def build_arch(self, arch):
        build_dir = join(self.get_build_dir(arch.arch), 'build')
        ensure_dir(build_dir)

        opencv_extras = []
        if 'opencv_extras' in self.ctx.recipe_build_order:
            opencv_extras_dir = self.get_recipe(
                'opencv_extras', self.ctx).get_build_dir(arch.arch)
            opencv_extras = [
                f'-DOPENCV_EXTRA_MODULES_PATH={opencv_extras_dir}/modules',
                '-DBUILD_opencv_legacy=OFF',
            ]

        with current_directory(build_dir):
            env = self.get_recipe_env(arch)
            python_major = self.ctx.python_recipe.version[0]
            python_include_root = self.ctx.python_recipe.include_root(arch.arch)
            python_site_packages = self.ctx.get_site_packages_dir(arch)
            python_link_root = self.ctx.python_recipe.link_root(arch.arch)
            python_link_version = self.ctx.python_recipe.link_version
            python_library = join(python_link_root,
                                  'libpython{}.so'.format(python_link_version))

            # ← 唯一改动：用 numpy recipe 的 get_include() 而非硬编码路径
            numpy_recipe = self.get_recipe('numpy', self.ctx)
            python_include_numpy = numpy_recipe.get_include(arch)

            shprint(sh.cmake,
                '-DP4A=ON',
                '-DANDROID_ABI={}'.format(arch.arch),
                '-DANDROID_STANDALONE_TOOLCHAIN={}'.format(self.ctx.ndk_dir),
                '-DANDROID_NATIVE_API_LEVEL={}'.format(self.ctx.ndk_api),
                '-DANDROID_EXECUTABLE={}/tools/android'.format(env['ANDROID_SDK']),
                '-DANDROID_SDK_TOOLS_VERSION=6514223',
                '-DANDROID_PROJECTS_SUPPORT_GRADLE=ON',
                '-DCMAKE_TOOLCHAIN_FILE={}'.format(
                    join(self.ctx.ndk_dir, 'build', 'cmake', 'android.toolchain.cmake')),
                '-DCMAKE_SHARED_LINKER_FLAGS=-L{path} -lpython{version}'.format(
                    path=python_link_root, version=python_link_version),
                '-DBUILD_WITH_STANDALONE_TOOLCHAIN=ON',
                '-DBUILD_SHARED_LIBS=ON',
                '-DBUILD_STATIC_LIBS=OFF',
                '-DBUILD_opencv_java=OFF',
                '-DBUILD_opencv_java_bindings_generator=OFF',
                '-DBUILD_TESTS=OFF',
                '-DBUILD_PERF_TESTS=OFF',
                '-DENABLE_TESTING=OFF',
                '-DBUILD_EXAMPLES=OFF',
                '-DBUILD_ANDROID_EXAMPLES=OFF',
                '-DBUILD_OPENCV_PYTHON{major}=ON'.format(major=python_major),
                '-DBUILD_OPENCV_PYTHON{major}=OFF'.format(
                    major='2' if python_major == '3' else '3'),
                '-DOPENCV_SKIP_PYTHON_LOADER=ON',
                '-DOPENCV_PYTHON{major}_INSTALL_PATH={site_packages}'.format(
                    major=python_major, site_packages=python_site_packages),
                '-DPYTHON_DEFAULT_EXECUTABLE={}'.format(self.ctx.hostpython),
                '-DPYTHON{major}_EXECUTABLE={host_python}'.format(
                    major=python_major, host_python=self.ctx.hostpython),
                '-DPYTHON{major}_INCLUDE_PATH={include_path}'.format(
                    major=python_major, include_path=python_include_root),
                '-DPYTHON{major}_LIBRARIES={python_lib}'.format(
                    major=python_major, python_lib=python_library),
                '-DPYTHON{major}_NUMPY_INCLUDE_DIRS={numpy_include}'.format(
                    major=python_major, numpy_include=python_include_numpy),
                '-DPYTHON{major}_PACKAGES_PATH={site_packages}'.format(
                    major=python_major, site_packages=python_site_packages),
                *opencv_extras,
                self.get_build_dir(arch.arch),
                _env=env)

            shprint(sh.make, '-j' + str(cpu_count()), 'opencv_python' + python_major)
            shprint(sh.cmake, '-DCOMPONENT=python', '-P', './cmake_install.cmake')
            sh.cp('-a', sh.glob('./lib/{}/lib*.so'.format(arch.arch)),
                  self.ctx.get_libs_dir(arch.arch))


recipe = OpenCVRecipe()
