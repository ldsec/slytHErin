package cipherUtils

import "C"
import (
	"fmt"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks/bootstrapping"
	"sync"
)

//Centralized Bootstrapping
func BootStrapBlocks(X *EncInput, Box CkksBox) {
	var wg sync.WaitGroup
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			wg.Add(1)
			go func(btp *bootstrapping.Bootstrapper, i, j int) {
				defer wg.Done()
				X.Blocks[i][j] = btp.Bootstrapp(X.Blocks[i][j])
			}(Box.BootStrapper.ShallowCopy(), i, j)
		}
	}
	wg.Wait()
	fmt.Println("Level after bootstrapping: ", X.Blocks[0][0].Level())
}

//Dummy Bootstrap where cipher is freshly encrypted
func DummyBootStrapBlocks(X *EncInput, Box CkksBox) *EncInput {
	pt := DecInput(X, Box)
	Xnew, err := NewEncInput(pt, X.RowP, X.ColP, Box.Params.MaxLevel(), Box)
	utils.ThrowErr(err)
	return Xnew
}

//Deprecated
func RescaleBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			Box.Evaluator.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
		}
	}
}

//Deprecated
func RemoveImagFromBlocks(X *EncInput, Box CkksBox) {
	for i := 0; i < X.RowP; i++ {
		for j := 0; j < X.ColP; j++ {
			Box.Evaluator.MultByConst(X.Blocks[i][j], 0.5, X.Blocks[i][j])
			Box.Evaluator.Add(X.Blocks[i][j], Box.Evaluator.ConjugateNew(X.Blocks[i][j]), X.Blocks[i][j])
			//Box.Evaluator.Rescale(X.Blocks[i][j], Box.Params.DefaultScale(), X.Blocks[i][j])
		}
	}
	RescaleBlocks(X, Box)
}
