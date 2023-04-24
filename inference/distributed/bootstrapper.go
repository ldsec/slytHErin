package distributed

import "github.com/ldsec/slytHErin/inference/cipherUtils"

// distributed bootstrapper
type DistributedBtp struct {
	master   *LocalMaster
	minLevel int
}

func NewDistributedBootstrapper(master *LocalMaster, minLevel int) *DistributedBtp {
	return &DistributedBtp{master: master, minLevel: minLevel}
}

// Starts refresh protocol with master
func (Btp *DistributedBtp) Bootstrap(X *cipherUtils.EncInput, Box cipherUtils.CkksBox) {
	Btp.master.StartProto(REFRESH, X, nil, Btp.minLevel, Box)
}
