#!/bin/bash

# Set the source file to be copied
source_file="../../data_bases/BrLib_MIR.csv"

# Define a function to copy the file to a data_bases directory
copy_to_data_bases() {
  local data_bases_dir="$1"
  #local destination_dir="$data_bases_dir/data_bases_copy"
        
  # Create a new directory if it doesn't exist
  #mkdir -p "$destination_dir"
              
  # Copy the source file to the destination directory
  cp "$source_file" "$data_bases_dir"
}

# Find and copy to data_bases directories
find . -type d -name "data_bases" | while read -r data_bases_dir; do
                                      copy_to_data_bases "$data_bases_dir"
                                    done

echo "File copied to all data_bases directories."

#  find .: This part of the command initiates the find utility, which is used to search for files and directories.
#
# -type d: This option instructs find to look for directories. It filters the search to only consider directories.
#
# -name "data_bases": This option specifies that we are looking for directories with the name "data_bases." It's a case-sensitive search for directories with the exact name "data_bases."
#
# |: This is a pipe symbol, which takes the output from the find command and passes it as input to the next command.
#
# while read -r data_bases_dir; do: This is the start of a loop that reads each directory found by the find command and assigns it to the variable data_bases_dir.
#
# copy_to_data_bases "$data_bases_dir": Within the loop, this line calls the copy_to_data_bases function, passing the directory path as an argument. It's responsible for copying the source file into the current data_bases directory.
#
# done: This marks the end of the loop, and the loop continues until all data_bases directories are processed.
