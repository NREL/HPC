#!/bin/bash

CONTAINER=""
DIRECTORY=$(pwd)
ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--container)
      CONTAINER=$2
      shift
      shift
      ;;
    -d|--directory)
      DIRECTORY=$2
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [ -d ${DIRECTORY} ]; then
    echo "Error: ${DIRECTORY} does not exist"
    exit 1
fi

if [ $CONTAINER != "" ]; then
    if ! [ -f ${CONTAINER} ]; then
        echo "Error: container=${CONTAINER} does not exist"
        exit 1
    fi
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_DIR=$(dirname ${SCRIPT_DIR})/conf
CONFIG_FILE=$(dirname ${SCRIPT_DIR})/config
if [ -z ${CONFIG_DIR} ]; then
    echo "Error: the directory ${CONFIG_DIR} does not exist"
    exit 1
fi
if [ -z ${CONFIG_FILE} ]; then
    echo "Error: the file ${CONFIG_FILE} does not exist"
    exit 1
fi
cp -r ${CONFIG_DIR} ${DIRECTORY}
cp ${CONFIG_FILE} ${DIRECTORY}

if [ $CONTAINER != "" ]; then
    CONTAINER=${CONTAINER//\//\\\/}
    sed -i "s/container = .*/container = ${CONTAINER}/" ${DIRECTORY}/config
fi

