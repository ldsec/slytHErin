package cipherUtils

import "C"
import (
	"github.com/ldsec/slytHErin/inference/plainUtils"
	"github.com/ldsec/slytHErin/inference/utils"
	"sync"
)

// Interface for bootstrappers
type IBootstrapper interface {
	Bootstrap(input *EncInput, Box CkksBox)
}

// Centralized bootstrapper. Homomorphically evaluates decryption circuit
type Bootstrapper struct {
	poolSize int
}

func NewBootstrapper(poolSize int) *Bootstrapper {
	Btp := new(Bootstrapper)
	Btp.poolSize = poolSize
	return Btp
}

func (Btp *Bootstrapper) spawnEvaluators(X *EncInput, ch chan []int, Box CkksBox) {
	btp := Box.BootStrapper.ShallowCopy()
	for {
		coords, ok := <-ch //feed the goroutines
		if !ok {
			//if channel is closed
			return
		}
		i, j := coords[0], coords[1]
		X.Blocks[i][j] = btp.Bootstrapp(X.Blocks[i][j])
	}
}

// Centralized Bootstrapping
func (Btp *Bootstrapper) Bootstrap(X *EncInput, Box CkksBox) {

	if Btp.poolSize == 1 {
		//single threaded
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				X.Blocks[i][j] = Box.BootStrapper.Bootstrapp(X.Blocks[i][j])
			}
		}
	} else if Btp.poolSize > 1 {
		//bounded threading

		ch := make(chan []int)
		var wg sync.WaitGroup
		//spawn consumers
		for i := 0; i < Btp.poolSize; i++ {
			wg.Add(1)
			go func() {
				Btp.spawnEvaluators(X, ch, Box)
				defer wg.Done()
			}()
		}
		//feed consumers
		for i := 0; i < X.RowP; i++ {
			for j := 0; j < X.ColP; j++ {
				ch <- []int{i, j}
			}
		}
		close(ch)
		wg.Wait()
	}
}

// Dummy Bootstrap where cipher is freshly encrypted
func DummyBootStrapBlocks(X *EncInput, Box CkksBox) *EncInput {
	pt := DecInput(X, Box)
	Xnew, err := NewEncInput(plainUtils.NewDense(pt), X.RowP, X.ColP, Box.Params.MaxLevel(), Box.Params.DefaultScale(), Box)
	utils.ThrowErr(err)
	return Xnew
}
