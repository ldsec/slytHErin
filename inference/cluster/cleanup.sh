#!/bin/bash

#Stops process on remote nodes after remote.sh

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Stops process on remote nodes after remote.sh"
   echo "Make sure that you have already the servers in the ~/.ssh/known_hosts file"
   echo "Syntax: setup.sh [-options]"
   echo "options:"
   echo "p     parties to stop, 2 for client-server protocol or #servers for distributed scenario"
   echo "h     help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
partiesOpt=0
while getopts ":p:h" flag; do
    case "${flag}" in
        p) partiesOpt=${OPTARG};;
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
  id=$(jq ".cluster_ids[${i}]" < $FILE)
  if [ $id -lt 100 ]; then
    id="0""$id"
  fi
  ip=$(jq ".cluster_ips[${i}]" < $FILE)
  echo "connecting to: ${user}:${pwd}@iccluster${id}.iccluster.epfl.ch at address ${ip}"
  sshpass -p 1 ssh -o StrictHostKeyChecking=no root@iccluster"${id}".iccluster.epfl.ch "pkill -f 'inference'" &
  i=$((i+1))
done

