#!/bin/bash
set -euo pipefail

pushd slicer-src

perl -0pi -e 's#https://www\.mpfr\.org/mpfr-current/mpfr-4\.2\.1\.tar\.bz2#https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.bz2#g' deps/MPFR/MPFR.cmake
perl -0pi -e 's#URL_HASH SHA256=c56edfacef0a60c0de3e6489194fcb2f24c03dbb550a8a7de5938642d045bd32#URL_HASH SHA256=17a3e875acece9be40b093361cfef47385d4ef22c995ffbf36b2871f5785f9b8#g' deps/TIFF/TIFF.cmake

sudo apt-get install -y libgtk-3-dev
if [[ -f "build_linux.sh" ]]; then
  ./build_linux.sh -u
else
  ./BuildLinux.sh -u
fi

popd
