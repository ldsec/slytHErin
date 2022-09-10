#!/bin/bash

#Moves config file with sftp to remote nodes and starts them

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Starts the players on remote servers -- run after setup.sh!"
   echo "Make sure that you have already the servers in the ~/.ssh/known_hosts file"
   echo "Make sure you have the file config.json in the same folder"
   echo "Syntax: setup.sh [-options]"
   echo "options:"
   echo "l     logN value, 14 or 15"
   echo "m     model, crypto or nn"
   echo "z     nn value, 20 or 50"
   echo "h     help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
while getopts ":l:m:z:h" flag; do
    case "${flag}" in
        l) logN=${OPTARG};;
        m) model=${OPTARG};;
        z) nn=${OPTARG};;
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
while [ $i -lt $parties ]; do
  id=$(jq ".cluster_ids[${i}]" < $FILE)
  if [ $id -lt 100 ]; then
    id="0""$id"
  fi
  ip=$(jq ".cluster_ips[${i}]" < $FILE)
  echo "connecting to: ${user}:${pwd}@iccluster${id}.iccluster.epfl.ch at address ${ip}"
  sshpass -p 1 ssh root@iccluster"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x inference; ./inference --nn ${nn} --model ${model} --logN ${logN} --addr ${ip};" &
  i=$((i+1))
done