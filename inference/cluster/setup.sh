#!/bin/bash

#Moves config file with sftp to remote nodes

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Make sure that you have already the servers in the ssh known hosts"
   echo "Make sure that you have config.json in same folder"
   echo "Syntax: setup.sh [-h]"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

while getopts ":h" flag; do
    case "${flag}" in
        h) Help
           exit;;
    esac
done

FILE=config.json
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    exit
fi
parties=$(jq ".num_servers" < $FILE)
user=$(jq ".ssh_user" < $FILE)
user=$(echo "$user" | tr -d '"')
pwd=$(jq ".ssh_pwd" < $FILE)
pwd=$(echo "$pwd" | tr -d '"')

i=0
cwd=$(pwd)
while [ $i -lt $parties ]; do
  id=$(jq ".cluster_ids[${i}]" < $FILE)
  if [ $id -lt 100 ]; then
      id="0""$id"
  fi
  echo "connecting to: '${user}:${pwd}@iccluster${id}.iccluster.epfl.ch'"
  lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "rmdir dnn; mkdir dnn; cd dnn; rmdir config; mkdir config; exit"
  cd ../../config
  for f in *; do
    lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "cd /root/dnn/config; put ${f}; exit"
  done
  sshpass -p "${pwd}" ssh "${user}"@iccluster"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x go_install.sh; ./go_install.sh;"
  i=$((i+1))
  cd "$cwd"
done
