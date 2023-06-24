#!/bin/bash

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

resolve_hopswork() {
  # Check if librdkafka is installed and at the correct version
  installed_versions=$(brew list --versions librdkafka)
  required_version="1.9.2"  # replace this with the version you want

  if ! echo "$installed_versions" | grep -q "$required_version"; then
    # If librdkafka is not installed or not at the correct version, proceed with installation

    # see https://community.hopsworks.ai/t/ssl-handshake-failed-on-macos-hopsworks-serverless/886/3
    curl -O https://raw.githubusercontent.com/Homebrew/homebrew-core/f7d0f40bbc4075177ecf16812fd95951a723a996/Formula/librdkafka.rb
    brew install --build-from-source librdkafka.rb
    rm librdkafka.rb
  else
    logger "INFO" "librdkafka is already installed at version: $(brew list --versions librdkafka)"
  fi

  # Set VERSION to the required version, assuming it is now installed
  VERSION=$required_version
  # use below if the librdkafka version is fixed
  # VERSION=$(ls /opt/homebrew/Cellar/librdkafka | tail -n 1)

  # Export necessary environment variables
  export C_INCLUDE_PATH=/opt/homebrew/Cellar/librdkafka/$VERSION/include
  export LIBRARY_PATH=/opt/homebrew/Cellar/librdkafka/$VERSION/lib
}

resolve_lightgbm() {
  # see https://stackoverflow.com/questions/74566704/cannot-install-lightgbm-3-3-3-on-apple-silicon
  # Check if cmake is installed
  if ! brew list --versions cmake >/dev/null; then
    # If cmake is not installed, install it
    brew install cmake
  else
    logger "INFO" "cmake is already installed at version: $(cmake --version)"
  fi

  # Check if libomp is installed
  if ! brew list --versions libomp >/dev/null; then
    # If libomp is not installed, install it
    brew install libomp
  else
    logger "INFO" "libomp is already installed at version: $(brew list --versions libomp)"
  fi
}


custom_install_hopswork_and_lightgbm_if_arm64() {
    # Check if on macOS with M1 or ARM chip
    if [[ "$(uname -m)" == "arm64" ]]; then
        logger "INFO" "Installing librdkafka for M1 chip"
        resolve_hopswork
        resolve_lightgbm
    fi
}

custom_install_hopswork_and_lightgbm_if_arm64