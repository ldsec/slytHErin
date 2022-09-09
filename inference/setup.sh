#!/bin/bash

#Moves config file with sftp to remote nodes

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Make sure that you have already the servers in the ssh known hosts"
   echo "Syntax: setup.sh [-options]"
   echo "options:"
   echo "c     initial id of first server in cluster, e.g is id goes from 76 to 78 it's 76"
   echo "n     num of parties"
   echo "u     ssh username"
   echo "p     ssh password"
   echo "h     help"
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

while getopts ":c:n:u:p:h" flag; do
    case "${flag}" in
        c) startingId=${OPTARG};;
        n) parties=${OPTARG};;
        u) username=${OPTARG};;
        p) password=${OPTARG};;
        h) Help
           exit;;
    esac
done
echo "startingId: $startingId";
echo "parties: $parties";
echo "username: $username";
echo "password: $password";

i=0
cd config
while [ $i -lt $parties ]; do
  id=$((startingId+i))
  echo "connecting to: '${username}:${password}@iccluster0${id}.iccluster.epfl.ch'"
  lftp sftp://"${username}":"${password}"@iccluster0"${id}".iccluster.epfl.ch -e "rmdir dnn; mkdir dnn; cd dnn; rmdir config; mkdir config; cd config; exit"
  for FILE in *; do
    lftp sftp://"${username}":"${password}"@iccluster0"${id}".iccluster.epfl.ch -e "cd dnn; cd config; put ${FILE}; exit"
  done
  sshpass -p 1 ssh root@iccluster0"${id}".iccluster.epfl.ch "cd /root/dnn/config; chmod +x go_install.sh; ./go_install.sh;"
  i=$((i+1))
done
