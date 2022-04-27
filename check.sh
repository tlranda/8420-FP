#!/bin/bash

qstat -u ${USER} | tail -n +2;
echo;
files=`ls -l src/transformers/saved/ | tail -n +2 | sed 's/[[:blank:]]*$//;s/.*[[:blank:]]//' | grep -v -e "attempt" -e "one_epoch" -e "testepoch"`;
if [[ ${#files} -eq 0 ]]; then
  echo "No saved files";
else
  for f in ${files[@]}; do
    echo $f;
  done;
fi

