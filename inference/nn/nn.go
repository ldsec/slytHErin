package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/distributed"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/rlwe"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"os"
	"sync"
	"time"
)

/*
	Stores NN model in clear, as in json format
*/
type NN struct {
	Conv   utils.Layer   `json:"conv"`
	Dense  []utils.Layer `json:"dense"`
	Layers int           `json:"layers"`

	RowsOutConv, ColsOutConv, ChansOutConv, DimOutDense int //dimentions
}

/*
	Wrapper for Plaintext layers in Block Matrix form
*/
type NNBlock struct {
	Weights              []*plainUtils.BMatrix
	Bias                 []*plainUtils.BMatrix
	InnerRows, InnerCols []int //inner dim of sub-matrices
	RowsP, ColsP         []int //partition of BMatrix

	ActName    string
	Activation func(*plainUtils.BMatrix) *plainUtils.BMatrix
	Layers     int
}

/*
	Wrapper for Encrypted layers in Block Matrix form
*/
type NNEnc struct {
	Weights    []*cipherUtils.EncWeightDiag
	Bias       []*cipherUtils.EncInput
	Activators []*cipherUtils.Activator

	ReLUApprox *utils.ChebyPolyApprox //this will store the coefficients of the poly approximating ReLU
	Box        cipherUtils.CkksBox
	Layers     int
}

//Approximation parameters for Chebychev approximated activations. Depends on the number of layers
type ApproxParams struct {
	a, b float64
	deg  int
}

var NN20Params = ApproxParams{a: -30, b: 30, deg: 63}
var NN20Params_CentralizedBtp = ApproxParams{a: -30, b: 30, deg: 15} //deg needs to be < residual capacity
var NN50Params = ApproxParams{a: -55, b: 55, deg: 63}

//computes how many levels are needed to complete the pipeline
func (nne NNEnc) LevelsToComplete(currLayer int, afterMul bool) int {
	levelsNeeded := 0
	for i := currLayer; i < nne.Layers+1; i++ {
		levelsNeeded += 1 //mul
		if i != nne.Layers {
			//last layer with no act
			levelsNeeded += nne.ReLUApprox.LevelsOfAct()
		}
	}
	if afterMul {
		levelsNeeded--
	}
	//fmt.Printf("Levels needed from layer %d to complete: %d\n\n", currLayer+1, levelsNeeded)
	return levelsNeeded
}

// loads json file with weights
func LoadNN(path string) *NN {

	jsonFile, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)

	var res NN
	json.Unmarshal([]byte(byteValue), &res)
	return &res
}

func (nn *NN) Init(layers int) {
	//init dimensional values (not really used, just for reference)
	nn.Layers = layers
	//nn.RowsOutConv = 21
	//nn.ColsOutConv = 20
	//nn.ChansOutConv = 2 //tot is 21*20*2 = 840 per sample after conv
	//nn.DimOutDense = 92 //per sample, all dense are the same but last one which is 92x10
}

//Builds BlockMatrices from the layers in cleartext
func (nn *NN) NewBlockNN(batchSize, InRowP, InColP int) (*NNBlock, error) {
	rowP, colP := InRowP, InColP
	layers := nn.Layers
	//Assemble layers in block matrix form
	convM := utils.BuildKernelMatrix(nn.Conv.Weight)
	//convMB, _ := plainUtils.PartitionMatrix(convM, colP, 28) //900x840 --> submatrices are 30x30
	//use if model from go training
	convMB, _ := plainUtils.PartitionMatrix(convM, colP, 26) //784x676 --> subm 28x26
	biasConvM := utils.BuildBiasMatrix(nn.Conv.Bias, plainUtils.NumCols(convM), batchSize)
	biasConvMB, _ := plainUtils.PartitionMatrix(biasConvM, rowP, convMB.ColP)

	denseMatrices := make([]*mat.Dense, layers)
	denseBiasMatrices := make([]*mat.Dense, layers)
	denseMatricesBlock := make([]*plainUtils.BMatrix, layers)
	denseBiasMatricesBlock := make([]*plainUtils.BMatrix, layers)

	for i := 0; i < layers; i++ {
		denseMatrices[i] = utils.BuildKernelMatrix(nn.Dense[i].Weight)
		denseBiasMatrices[i] = utils.BuildBiasMatrix(nn.Dense[i].Bias, plainUtils.NumCols(denseMatrices[i]), batchSize)
		if i == 0 {
			//840x92 --> 30x23
			//denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 28, 4)
			//if go training 676x92 --> 26x23
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 26, 4)
		} else if i == layers-1 {
			//92x10 --> 23x10
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 4, 1)
		} else {
			//92x92 --> 23x23
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 4, 4)
		}
		denseBiasMatricesBlock[i], _ = plainUtils.PartitionMatrix(
			denseBiasMatrices[i],
			rowP,
			denseMatricesBlock[i].ColP)
	}

	weightMatrices := []*mat.Dense{convM}
	weightMatrices = append(weightMatrices, denseMatrices...)
	biasMatrices := []*mat.Dense{biasConvM}
	biasMatrices = append(biasMatrices, denseBiasMatrices...)

	weightMatricesBlock := []*plainUtils.BMatrix{convMB}
	weightMatricesBlock = append(weightMatricesBlock, denseMatricesBlock...)
	biasMatricesBlock := []*plainUtils.BMatrix{biasConvMB}
	biasMatricesBlock = append(biasMatricesBlock, denseBiasMatricesBlock...)

	//Collects Partitions and Dimentions of blocks
	rowsW := make([]int, len(weightMatricesBlock))
	colsW := make([]int, len(weightMatricesBlock))
	rowsPW := make([]int, len(weightMatricesBlock))
	colsPW := make([]int, len(weightMatricesBlock))
	for w := range weightMatricesBlock {
		rowsW[w], colsW[w] = weightMatricesBlock[w].InnerRows, weightMatricesBlock[w].InnerCols
		rowsPW[w], colsPW[w] = weightMatricesBlock[w].RowP, weightMatricesBlock[w].ColP
	}
	actName := "soft relu"
	activation := func(matrix *plainUtils.BMatrix) *plainUtils.BMatrix {
		for i := 0; i < matrix.RowP; i++ {
			for j := 0; j < matrix.ColP; j++ {
				for ii := 0; ii < matrix.InnerRows; ii++ {
					for jj := 0; jj < matrix.InnerCols; jj++ {
						//soft relu
						matrix.Blocks[i][j].Set(ii, jj, utils.SoftReLu(matrix.Blocks[i][j].At(ii, jj)))
					}
				}
			}
		}
		return matrix
	}

	return &NNBlock{
		Weights:    weightMatricesBlock,
		Bias:       biasMatricesBlock,
		InnerRows:  rowsW,
		InnerCols:  colsW,
		RowsP:      rowsPW,
		ColsP:      colsPW,
		ActName:    actName,
		Activation: activation,
		Layers:     layers,
	}, nil
}

//Forms an encrypted NN from the plaintext representation
func (nnb *NNBlock) NewEncNN(batchSize, InRowP, btpCapacity int, Box cipherUtils.CkksBox, minLevel int) (*NNEnc, error) {
	layers := nnb.Layers
	innerRows := batchSize / InRowP
	nne := new(NNEnc)
	nne.Weights = make([]*cipherUtils.EncWeightDiag, layers+1)
	nne.Bias = make([]*cipherUtils.EncInput, layers+1)
	nne.Activators = make([]*cipherUtils.Activator, layers)
	nne.Layers = nnb.Layers
	if layers == 20 {
		if minLevel != -1 {
			//distributed
			nne.ReLUApprox = utils.InitActivationCheby(nnb.ActName, NN20Params.a, NN20Params.b, NN20Params.deg)
		} else {
			nne.ReLUApprox = utils.InitActivationCheby(nnb.ActName, NN20Params_CentralizedBtp.a, NN20Params_CentralizedBtp.b, NN20Params_CentralizedBtp.deg)
		}
	} else if layers == 50 {
		nne.ReLUApprox = utils.InitActivationCheby(nnb.ActName, NN50Params.a, NN50Params.b, NN50Params.deg)
	}
	maxLevel := Box.Params.MaxLevel()
	level := maxLevel

	fmt.Println("Creating weights encrypted block matrices...")

	for i := 0; i < layers+1; i++ {
		if ((level <= minLevel && minLevel != -1) || level == 0) && level < nne.LevelsToComplete(i, false) {
			//bootstrap
			if minLevel != -1 {
				//distributed
				if level < minLevel {
					s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
					utils.ThrowErr(errors.New(s))
				}
				level = maxLevel
			} else {
				//centralized
				level = btpCapacity
			}
		}
		//change to cheby base
		a := nne.ReLUApprox.A
		b := nne.ReLUApprox.B
		mulC := 2 / (b - a)
		addC := (-a - b) / (b - a)
		if i == layers {
			//skip base switch for activation, since there is none in last layer
			mulC = 1.0
			addC = 0.0
		}
		nne.Weights[i], _ = cipherUtils.NewEncWeightDiag(
			plainUtils.MatToArray(plainUtils.MulByConst(plainUtils.ExpandBlocks(nnb.Weights[i]),
				mulC)),
			nnb.RowsP[i], nnb.ColsP[i], batchSize, level, Box)

		level-- //mul
		if (level < minLevel && level < nne.LevelsToComplete(i, true)) || level < 0 {
			if minLevel > 0 {
				panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
			} else {
				panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
			}
		}

		nne.Bias[i], _ = cipherUtils.NewEncInput(
			plainUtils.MatToArray(plainUtils.AddConst(plainUtils.MulByConst(plainUtils.ExpandBlocks(nnb.Bias[i]),
				mulC), addC)),
			InRowP, nnb.ColsP[i], level, Box)
		if (level < nne.ReLUApprox.LevelsOfAct() || (minLevel != -1 && (level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel))) && level < nne.LevelsToComplete(i, true) {
			if minLevel != -1 {
				//distributed
				if level < minLevel {
					s := fmt.Sprintf("Estimated level below minlevel for layer %d", i+1)
					utils.ThrowErr(errors.New(s))
				}
				level = maxLevel
			} else {
				//centralized
				level = btpCapacity
			}
		}
		if i != layers {
			var err error
			nne.Activators[i], err = cipherUtils.NewActivator(nne.ReLUApprox, level, Box.Params.DefaultScale(), innerRows, nne.Weights[i].InnerCols, InRowP, nne.Weights[i].ColP, Box)
			utils.ThrowErr(err)
			level -= nne.ReLUApprox.LevelsOfAct() //activation
			if (level < minLevel && level < nne.LevelsToComplete(i+1, true)) || level < 0 {
				if minLevel > 0 {
					panic(errors.New(fmt.Sprintf("Level below minimum level at layer %d\n", i+1)))
				} else {
					panic(errors.New(fmt.Sprintf("Level below 0 at layer %d\n", i+1)))
				}
			}
		}
	}
	fmt.Println("Done...")
	return nne, nil
}

func (nnb *NNBlock) EvalBatchPlain(XBatchClear *plainUtils.BMatrix, Y []int, labels int) (int, time.Duration) {
	var err error
	XintPlain := XBatchClear
	now := time.Now()
	for i := range nnb.Weights {
		XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
		utils.ThrowErr(err)
		XintPlain, err = plainUtils.AddBlocks(XintPlain, nnb.Bias[i])
		utils.ThrowErr(err)
		XintPlain = nnb.Activation(XintPlain)
	}
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	batchSize := XBatchClear.RowP * XBatchClear.InnerRows
	resPlain := plainUtils.MatToArray(plainUtils.ExpandBlocks(XintPlain))
	predictionsPlain := make([]int, batchSize)

	correctsPlain := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := resPlain[i][j]
			if confidence > maxConfidence {
				maxConfidence = confidence
				maxIdx = j
			}
		}
		predictionsPlain[i] = maxIdx
		if predictionsPlain[i] == Y[i] {
			correctsPlain += 1
		}
	}
	fmt.Println("Corrects clear:", correctsPlain)
	return correctsPlain, elapsed
}

func (nne *NNEnc) EvalBatchEncrypted(nnb *NNBlock, XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput, Box cipherUtils.CkksBox, labels int, debug bool) (int, time.Duration) {
	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	now := time.Now()
	for i := range nne.Weights {
		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == 0 {
			fmt.Println("Level 0, Bootstrapping...")
			fmt.Println("pre boot")
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			cipherUtils.BootStrapBlocks(Xint, Box)
			fmt.Println("after boot")
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}
		Xint, err = cipherUtils.BlocksC2CMul(Xint, nne.Weights[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
			utils.ThrowErr(err)
			fmt.Printf("Mul ")
			//cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nnb.ReLUApprox.Interval), Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}

		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, nne.Bias[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.AddBlocks(XintPlain, nnb.Bias[i])
			utils.ThrowErr(err)
		}
		//activation
		if i != len(nne.Weights)-1 {
			level = Xint.Blocks[0][0].Level()
			if level < nne.ReLUApprox.LevelsOfAct() {
				fmt.Println("Bootstrapping for Activation")
				fmt.Println("pre boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
				cipherUtils.BootStrapBlocks(Xint, Box)
				fmt.Println("after boot")
				//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
			fmt.Printf("Activation ")
			//cipherUtils.EvalPolyBlocks(Xint, nne.ReLUApprox.Poly, Box)
			nne.Activators[i].ActivateBlocks(Xint)
			if debug {
				XintPlain = nnb.Activation(XintPlain)
				//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
				cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
		}
	}
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	batchSize := XBatchClear.RowP * XBatchClear.InnerRows
	res := cipherUtils.DecInput(Xint, Box)
	resPlain := plainUtils.MatToArray(plainUtils.ExpandBlocks(XintPlain))
	predictions := make([]int, batchSize)
	predictionsPlain := make([]int, batchSize)

	corrects := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := res[i][j]
			if confidence > maxConfidence {
				maxConfidence = confidence
				maxIdx = j
			}
		}
		predictions[i] = maxIdx
		if predictions[i] == Y[i] {
			corrects += 1
		}
	}
	if debug {
		correctsPlain := 0
		for i := 0; i < batchSize; i++ {
			maxIdx := 0
			maxConfidence := 0.0
			for j := 0; j < labels; j++ {
				confidence := resPlain[i][j]
				if confidence > maxConfidence {
					maxConfidence = confidence
					maxIdx = j
				}
			}
			predictionsPlain[i] = maxIdx
			if predictionsPlain[i] == Y[i] {
				correctsPlain += 1
			}
		}
		fmt.Println("Corrects clear:", correctsPlain)
	}
	fmt.Println("Corrects enc:", corrects)
	return corrects, elapsed
}

func EvalBatchEncryptedDistributedDummy(nne *NNEnc, nnb *NNBlock,
	XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput,
	Box cipherUtils.CkksBox, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int, labels int, debug bool,
	master *distributed.DummyMaster) (int, int, time.Duration) {

	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	var wg sync.WaitGroup
	fmt.Println("Minlevel: ", minLevel)
	fmt.Println("MaxLevel: ", Box.Params.MaxLevel())
	now := time.Now()
	for i := range nne.Weights {
		fmt.Printf("======================> Layer %d\n", i+1)

		level := Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		if level <= minLevel { //minLevel for Bootstrapping
			if level < minLevel {
				utils.ThrowErr(errors.New("Below minLevel for bootstrapping"))
			}
			fmt.Println("MinLevel, Bootstrapping...")

			for ii := 0; ii < Xint.RowP; ii++ {
				for jj := 0; jj < Xint.ColP; jj++ {
					//parallel bootstrapping of all blocks
					//fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
					wg.Add(1)
					go func(ii, jj int, eval ckks.Evaluator) {
						defer wg.Done()
						eval.DropLevel(Xint.Blocks[ii][jj], Xint.Blocks[ii][jj].Level()-minLevel)
						Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[1], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
						utils.ThrowErr(err)
					}(ii, jj, Box.Evaluator.ShallowCopy())
				}
			}
			wg.Wait()
			fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())

		}

		Xint, err = cipherUtils.BlocksC2CMul(Xint, nne.Weights[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
			utils.ThrowErr(err)
			fmt.Println("Multiplication")
			//cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nnb.ReLUApprox.Interval), Box)
			cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}
		level = Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, nne.Bias[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.AddBlocks(XintPlain, nnb.Bias[i])
			utils.ThrowErr(err)
		}

		level = Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		if i != len(nne.Weights)-1 {
			if level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox.LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox.LevelsOfAct())
				} else if level == minLevel {
					fmt.Println("Min Level , Bootstrapping...")
				} else {
					fmt.Println("Activation would set level below threshold, Pre-emptive Bootstraping...")
					fmt.Println("Curr level: ", level)
					fmt.Println("Drop to: ", minLevel)
					fmt.Println("Diff: ", level-minLevel)
				}

				for ii := 0; ii < Xint.RowP; ii++ {
					for jj := 0; jj < Xint.ColP; jj++ {
						//parallel bootstrapping of all blocks
						//fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
						wg.Add(1)
						go func(ii, jj int, eval ckks.Evaluator) {
							defer wg.Done()
							eval.DropLevel(Xint.Blocks[ii][jj], Xint.Blocks[ii][jj].Level()-minLevel)
							Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[1], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
							utils.ThrowErr(err)
						}(ii, jj, Box.Evaluator.ShallowCopy())
					}
				}
				wg.Wait()

				fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())
			}

			fmt.Println("Activation")
			cipherUtils.EvalPolyBlocks(Xint, nne.ReLUApprox.Poly, Box)
			//nne.Activators[i].ActivateBlocks(Xint)
			if debug {
				XintPlain = nnb.Activation(XintPlain)
				//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
				cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
		}
	}
	fmt.Println("Key Switch to querier secret key")
	for ii := 0; ii < Xint.RowP; ii++ {
		for jj := 0; jj < Xint.ColP; jj++ {
			//parallel key switching
			fmt.Println("Switch id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
			wg.Add(1)
			go func(ii, jj int) {
				defer wg.Done()
				Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[0], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
				utils.ThrowErr(err)
			}(ii, jj)
		}
	}
	wg.Wait()
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	batchSize := XBatchClear.RowP * XBatchClear.InnerRows
	//use querier key to do the final decryption
	Box.Decryptor = decQ
	res := cipherUtils.DecInput(Xint, Box)
	resPlain := plainUtils.MatToArray(plainUtils.ExpandBlocks(XintPlain))
	predictions := make([]int, batchSize)
	predictionsPlain := make([]int, batchSize)

	corrects := 0
	correctsPlain := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := res[i][j]
			if confidence > maxConfidence {
				maxConfidence = confidence
				maxIdx = j
			}
		}
		predictions[i] = maxIdx
		if predictions[i] == Y[i] {
			corrects += 1
		}
	}
	if debug {
		for i := 0; i < batchSize; i++ {
			maxIdx := 0
			maxConfidence := 0.0
			for j := 0; j < labels; j++ {
				confidence := resPlain[i][j]
				if confidence > maxConfidence {
					maxConfidence = confidence
					maxIdx = j
				}
			}
			predictionsPlain[i] = maxIdx
			if predictionsPlain[i] == Y[i] {
				correctsPlain += 1
			}
		}
		fmt.Println("Corrects clear:", correctsPlain)
	}
	fmt.Println("Corrects enc:", corrects)
	return correctsPlain, corrects, elapsed
}

func EvalBatchEncryptedDistributedTCP(nne *NNEnc, nnb *NNBlock,
	XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput,
	Box cipherUtils.CkksBox, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int, labels int, debug bool,
	master *distributed.LocalMaster) (int, int, time.Duration) {

	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	var wg sync.WaitGroup
	fmt.Println("Minlevel: ", minLevel)
	fmt.Println("MaxLevel: ", Box.Params.MaxLevel())
	if debug {
		cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
	}
	now := time.Now()
	for i := range nne.Weights {
		fmt.Printf("======================> Layer %d\n", i+1)

		level := Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		if level <= minLevel && level < nne.LevelsToComplete(i, false) { //minLevel for Bootstrapping
			if level < minLevel {
				utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
			}
			fmt.Println("MinLevel, Bootstrapping...")
			for ii := 0; ii < Xint.RowP; ii++ {
				for jj := 0; jj < Xint.ColP; jj++ {
					//parallel bootstrapping of all blocks
					//fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
					wg.Add(1)
					go func(ii, jj int, eval ckks.Evaluator) {
						defer wg.Done()
						eval.DropLevel(Xint.Blocks[ii][jj], Xint.Blocks[ii][jj].Level()-minLevel)
						Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[1], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
						utils.ThrowErr(err)
					}(ii, jj, Box.Evaluator.ShallowCopy())
				}
			}
			wg.Wait()
			fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())
			if debug {
				fmt.Println("After bootstrapping")
				cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
		}

		Xint, err = cipherUtils.BlocksC2CMul(Xint, nne.Weights[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
			utils.ThrowErr(err)
			fmt.Println("Multiplication")
			//cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nnb.ReLUApprox.Interval), Box)
			cipherUtils.PrintDebugBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 2/(nne.ReLUApprox.B-nne.ReLUApprox.A)), Box)
		}
		level = Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, nne.Bias[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.AddBlocks(XintPlain, nnb.Bias[i])
			utils.ThrowErr(err)
		}

		level = Xint.Blocks[0][0].Level()
		fmt.Println("Ct level: ", level)
		//skip act

		if i != len(nne.Weights)-1 {
			if (level < nne.ReLUApprox.LevelsOfAct() || level <= minLevel || level-nne.ReLUApprox.LevelsOfAct() < minLevel) && level < nne.LevelsToComplete(i, true) {
				if level < minLevel {
					utils.ThrowErr(errors.New("level below minlevel for bootstrapping"))
				}
				if level < nne.ReLUApprox.LevelsOfAct() {
					fmt.Printf("Level < %d before activation , Bootstrapping...\n", nne.ReLUApprox.LevelsOfAct())
				} else if level == minLevel {
					fmt.Println("Min Level , Bootstrapping...")
				} else {
					fmt.Println("Activation would set level below threshold, Pre-emptive Bootstraping...")
					fmt.Println("Curr level: ", level)
					fmt.Println("Drop to: ", minLevel)
					fmt.Println("Diff: ", level-minLevel)
				}

				for ii := 0; ii < Xint.RowP; ii++ {
					for jj := 0; jj < Xint.ColP; jj++ {
						//parallel bootstrapping of all blocks
						//fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
						wg.Add(1)
						go func(ii, jj int, eval ckks.Evaluator) {
							defer wg.Done()
							eval.DropLevel(Xint.Blocks[ii][jj], Xint.Blocks[ii][jj].Level()-minLevel)
							Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[1], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
							utils.ThrowErr(err)
						}(ii, jj, Box.Evaluator.ShallowCopy())
					}
				}
				wg.Wait()

				fmt.Println("Level after bootstrapping: ", Xint.Blocks[0][0].Level())
				if debug {
					fmt.Println("After bootstrapping")
					cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
				}
			}

			fmt.Println("Activation")

			//cipherUtils.EvalPolyBlocks(Xint, nne.ReLUApprox.Poly, Box)
			nne.Activators[i].ActivateBlocks(Xint)
			if debug {
				XintPlain = nnb.Activation(XintPlain)
				//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
				cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			}
		}
	}
	fmt.Println("Key Switch to querier secret key")
	for ii := 0; ii < Xint.RowP; ii++ {
		for jj := 0; jj < Xint.ColP; jj++ {
			//parallel key switching
			fmt.Println("Switch id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
			wg.Add(1)
			go func(ii, jj int) {
				defer wg.Done()
				Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[0], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
				utils.ThrowErr(err)
			}(ii, jj)
		}
	}
	wg.Wait()
	elapsed := time.Since(now)
	fmt.Println("Done", elapsed)

	batchSize := XBatchClear.RowP * XBatchClear.InnerRows
	//use querier key to do the final decryption
	Box.Decryptor = decQ
	res := cipherUtils.DecInput(Xint, Box)
	resPlain := plainUtils.MatToArray(plainUtils.ExpandBlocks(XintPlain))
	predictions := make([]int, batchSize)
	predictionsPlain := make([]int, batchSize)

	corrects := 0
	correctsPlain := 0
	for i := 0; i < batchSize; i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < labels; j++ {
			confidence := res[i][j]
			if confidence > maxConfidence {
				maxConfidence = confidence
				maxIdx = j
			}
		}
		predictions[i] = maxIdx
		if predictions[i] == Y[i] {
			corrects += 1
		}
	}
	if debug {
		for i := 0; i < batchSize; i++ {
			maxIdx := 0
			maxConfidence := 0.0
			for j := 0; j < labels; j++ {
				confidence := resPlain[i][j]
				if confidence > maxConfidence {
					maxConfidence = confidence
					maxIdx = j
				}
			}
			predictionsPlain[i] = maxIdx
			if predictionsPlain[i] == Y[i] {
				correctsPlain += 1
			}
		}
		fmt.Println("Corrects clear:", correctsPlain)
	}
	fmt.Println("Corrects enc:", corrects)
	return correctsPlain, corrects, elapsed
}
