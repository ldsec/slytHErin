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
   echo "p     parties to start, 2 for client-server protocol or #servers for distributed scenario"
   echo "l     logN value, 14 or 15"
   echo "m     model, "crypto" or "nn""
   echo "z     nn value, 20 or 50"
   echo "h     help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
partiesOpt=0
nn=20
model="crypto"
logN=14
while getopts ":p:l:m:z:h" flag; do
    case "${flag}" in
        p) partiesOpt=${OPTARG};;
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

if [ $partiesOpt -gt 0 ]; then
  parties=$partiesOpt
else
  parties=$(jq ".num_servers" < $FILE)
fi
user=$(jq ".ssh_user" < $FILE)
user=$(echo "$user" | tr -d '"')
pwd=$(jq ".ssh_pwd" < $FILE)
pwd=$(echo "$pwd" | tr -d '"')

i=1 #first one is the master node / client
while [ $i -lt $parties ]; do
  ip=$(jq ".cluster_ips[${i}]" < $FILE)
  echo "connecting to: ${user}:${pwd} at ${ip}"
  sshpass -p 1 ssh -o StrictHostKeyChecking=no root@ip "cd /root/dnn/config; chmod +x inference; ./inference --nn ${nn} --model ${model} --logN ${logN} --addr ${ip};" &
  i=$((i+1))
done