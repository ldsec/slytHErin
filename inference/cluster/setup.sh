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
   echo "Syntax: setup.sh [-options]"
   echo "options:"
   echo "-l: y/n light version to move only inference binary with no data"
   echo "-h help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

light="n"

while getopts ":l:h" flag; do
    case "${flag}" in
        l) light=${OPTARG};;
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

echo "Light mode: $light"

i=0
cwd=$(pwd)
while [ $i -lt $parties ]; do
  id=$(jq ".cluster_ids[${i}]" < $FILE)
  if [ $id -lt 100 ]; then
      id="0""$id"
  fi
  echo "connecting to: '${user}:${pwd}@iccluster${id}.iccluster.epfl.ch'"

  if [ "$light" == "n" ]; then
    lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "rmdir dnn; mkdir dnn; cd dnn; rmdir config; mkdir config; exit"
  else
    lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "cd dnn/config; rm inference; exit"
  fi

  cd ../../config


  if [ "$light" == "n" ]; then
  # move all
    for f in *; do
      lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "cd /root/dnn/config; put ${f}; exit"
    done
    sshpass -p "${pwd}" ssh -o StrictHostKeyChecking=no "${user}"@iccluster"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x go_install.sh; ./go_install.sh;"
  else
    # move only inference binary
    lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "cd /root/dnn/config; put inference; exit"
    #lftp sftp://"${user}":"${pwd}"@iccluster"${id}".iccluster.epfl.ch -e "cd /root/dnn/config; put go_install.sh; exit"
    sshpass -p "${pwd}" ssh -o StrictHostKeyChecking=no "${user}"@iccluster"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x go_install.sh; ./go_install.sh;"
  fi
  i=$((i+1))
  cd "$cwd"
done
