#!/bin/bash
## https://codingfleet.com/code-converter/python/bash
## used to convert original python to bash

# Function to move specified paths to the front of a colon-separated list
upfront() {
  local p="$1"
  local move="$2"
  local myset=""
  local therest=""

  # Split the move string into individual paths
  IFS=, read -ra move <<< "$move"

  # Split the path string into individual paths
  IFS=: read -ra p <<< "$p"

  # Iterate through each move path
  for m in "${move[@]}"; do
    # Iterate through each path in the original path list
    for a in "${p[@]}"; do
      # Check if the move path is found in the current path
      if [[ "$a" =~ "$m" ]]; then
        # Append the path to the myset string
        myset="$myset$a:"
      else
        # Append the path to the therest string
        therest="$therest$a:"
      fi
    done
  done

  # Return the combined string with moved paths at the front
  echo "${myset}${therest%:}"
}

myrestore () {
  source /nopt/nrel/apps/env.sh

  local me="$USER"
  
  local p="$PATH"
  local np=$(upfront "$p" "$me")
  export PATH=$np
  
  local p="$LD_LIBRARY_PATH"
  local np=$(upfront "$p" "$me")
  export LD_LIBRARY_PATH=$np  
}
	
