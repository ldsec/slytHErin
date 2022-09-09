//Contains configuration variables for tests on the iccluster
package cluster

//ICCLUSTER - set accordingly before running tests!
//E.g: if we have 3 servers at 10.90.40.[2,3,4], subnet is 10.90.40 and starting addr is 2
//By convention, the master will be 10.90.40.2, the players 10.90.40.[3,4]
var SubNet = "10.90.40."
var StartingAddr = 2
var Parties = 3
