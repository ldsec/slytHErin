package distributed

import "github.com/ldsec/dnn-inference/inference/cipherUtils"

//distributed bootstrapper
type DistributedBtp struct {
	master   *LocalMaster
	minLevel int
}

func NewDistributedBootstrapper(master *LocalMaster, minLevel int) *DistributedBtp {
	return &DistributedBtp{master: master, minLevel: minLevel}
}

//Starts refresh protocol with master
func (Btp *DistributedBtp) Bootstrap(X *cipherUtils.EncInput) {
	Btp.master.StartProto(REFRESH, X, nil, Btp.minLevel)
}
