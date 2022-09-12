#!/bin/bash

#Grabs ip from cluster

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Grab ips from cluster -- run if config file has ip empty"
   echo "Make sure you have the file config.json in the same folder"
   echo "Syntax: setup.sh [-h]"
   echo "options:"
   echo "h     help"
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

cat config.json | jq ".cluster_ips = []" > tmp.json
mv tmp.json config.json

parties=$(jq ".num_servers" < $FILE)
user=$(jq ".ssh_user" < $FILE)
user=$(echo "$user" | tr -d '"')
pwd=$(jq ".ssh_pwd" < $FILE)
pwd=$(echo "$pwd" | tr -d '"')

echo "user: $user, pwd: $pwd, servers: $parties"

i=0
ips=()
while [ $i -lt $parties ]; do
  id=$(jq ".cluster_ids[${i}]" < $FILE)
  if [ $id -lt 100 ]; then
    id="0""$id"
  fi
  echo "connecting to: ${user}:${pwd}@iccluster${id}.iccluster.epfl.ch"

  ip=$(sshpass -p "${pwd}" ssh -o StrictHostKeyChecking=no "${user}"@iccluster"${id}".iccluster.epfl.ch hostname -I | tr -d " ")
  echo "ip is $ip"
  ips+=("$ip")
  i=$((i+1))
done

i=0
while [ $i -lt "$parties" ]; do
  cat config.json | jq ".cluster_ips += ["\""${ips[$i]}"\""]" > tmp.json
  mv tmp.json config.json
  i=$((i+1))
done


