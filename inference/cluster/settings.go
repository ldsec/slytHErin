//Contains configuration variables for tests on the iccluster
package cluster

import (
	"encoding/json"
	"github.com/ldsec/dnn-inference/inference/utils"
	"io/ioutil"
	"os"
)

//ICCLUSTER CONFIG

var configFile = "../cluster/config.json"

type Config struct {
	SshUser    string   `json:"ssh_user,omitempty"`
	SshPwd     string   `json:"ssh_password,omitempty"`
	NumServers int      `json:"num_servers,omitempty"`
	ClusterIds []int    `json:"cluster-ids,omitempty"`
	ClusterIps []string `json:"cluster_ips,omitempty"`
}

func ReadConfig(path string) *Config {
	jsonFile, err := os.Open(path)
	utils.ThrowErr(err)
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	c := new(Config)
	err = json.Unmarshal([]byte(byteValue), c)
	utils.ThrowErr(err)
	return c
}
