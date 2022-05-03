package nn

import (
	"encoding/json"
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
	"time"
)

/*
	Stores NN model in clear, right as in json format
*/
type NN struct {
	Conv       utils.Layer      `json:"conv"`
	Dense      []utils.Layer    `json:"dense"`
	Layers     int              `json:"layers"`
	ReLUApprox utils.PolyApprox //this will store the coefficients of the poly approximating ReLU

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

	ReLUApprox utils.PolyApprox //this will store the coefficients of the poly approximating ReLU
	Layers     int
}

/*
	Wrapper for Encrypted layers in Block Matrix form
*/
type NNEnc struct {
	Weights []*cipherUtils.EncWeightDiag
	Bias    []*cipherUtils.EncInput

	ReLUApprox utils.PolyApprox //this will store the coefficients of the poly approximating ReLU
	Box        cipherUtils.CkksBox
	Layers     int
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
	//init activation
	nn.ReLUApprox = utils.InitReLU()

	//init dimentional values (not really used, just for reference)
	nn.Layers = layers
	nn.RowsOutConv = 21
	nn.ColsOutConv = 20
	nn.ChansOutConv = 2 //tot is 21*20*2 = 840 per sample after conv
	nn.DimOutDense = 92 //per sample, all dense are the same but last one which is 92x10
}

//Builds BlockMatrices from the layers (cleartext)
func (nn *NN) NewBlockNN(batchSize, InRowP, InColP int) (*NNBlock, error) {
	rowP, colP := InRowP, InColP
	layers := nn.Layers
	//Assemble layers in block matrix form
	convM := utils.BuildKernelMatrix(nn.Conv.Weight)
	convMB, _ := plainUtils.PartitionMatrix(convM, colP, 28) //900x840 --> submatrices are 30x30
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
			denseMatricesBlock[i], _ = plainUtils.PartitionMatrix(denseMatrices[i], 28, 4)
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
	return &NNBlock{
		Weights:    weightMatricesBlock,
		Bias:       biasMatricesBlock,
		InnerRows:  rowsW,
		InnerCols:  colsW,
		RowsP:      rowsPW,
		ColsP:      colsPW,
		ReLUApprox: nn.ReLUApprox,
		Layers:     layers,
	}, nil
}

//Forms an encrypted NN from the plaintext repr
func (nnb *NNBlock) NewEncNN(batchSize, InRowP, btpCapacity int, Box cipherUtils.CkksBox) (*NNEnc, error) {
	layers := nnb.Layers
	nne := new(NNEnc)
	nne.Weights = make([]*cipherUtils.EncWeightDiag, layers+1)
	nne.Bias = make([]*cipherUtils.EncInput, layers+1)
	nne.Layers = nnb.Layers
	nne.ReLUApprox = nnb.ReLUApprox
	level := Box.Params.MaxLevel()
	fmt.Println("Creating weights encrypted block matrices...")
	for i := 0; i < layers+1; i++ {
		nne.Weights[i], _ = cipherUtils.NewEncWeightDiag(
			plainUtils.MatToArray(plainUtils.MulByConst(plainUtils.ExpandBlocks(nnb.Weights[i]),
				1.0/nnb.ReLUApprox.Interval)),
			nnb.RowsP[i], nnb.ColsP[i], batchSize, level, Box)
		level-- //mul
		nne.Bias[i], _ = cipherUtils.NewEncInput(
			plainUtils.MatToArray(plainUtils.MulByConst(plainUtils.ExpandBlocks(nnb.Bias[i]),
				1.0/nnb.ReLUApprox.Interval)),
			InRowP, nnb.ColsP[i], level, Box)
		level -= 2 //activation
		if level <= 0 {
			//lvl after btp is 2 --> #Qs-1 after StC
			level = btpCapacity
		}
	}
	fmt.Println("Done...")
	return nne, nil
}

//Deprecated
func (nn *NN) EvalBatchEncrypted(XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput, weightsBlock []*cipherUtils.EncWeightDiag, biasBlock []*cipherUtils.EncInput, weightsBlockPlain, biasBlockPlain []*plainUtils.BMatrix, Box cipherUtils.CkksBox, labels int) (int, time.Duration) {
	//pipeline
	now := time.Now()
	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	for i := range weightsBlock {
		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == 0 {
			fmt.Println("Level 0, Bootstrapping...")
			cipherUtils.BootStrapBlocks(Xint, Box)
		}
		Xint, err = cipherUtils.BlocksC2CMul(Xint, weightsBlock[i], Box)
		utils.ThrowErr(err)
		XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, weightsBlockPlain[i])
		utils.ThrowErr(err)
		fmt.Printf("Mul ")
		cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nn.ReLUApprox.Interval), Box)
		//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)

		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, biasBlock[i], Box)
		utils.ThrowErr(err)
		XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(biasBlockPlain[i], 1/nn.ReLUApprox.Interval))
		utils.ThrowErr(err)
		//fmt.Printf("Bias ")
		//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
		//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(biasBlockPlain[i], 1-1/nn.ReLUApprox.Interval))
		utils.ThrowErr(err)

		level = Xint.Blocks[0][0].Level()
		if level < 2 {
			//TO DO: ciphertext scale here complains
			fmt.Println("Bootstrapping for Activation")
			cipherUtils.BootStrapBlocks(Xint, Box)
		}
		//activation
		//TO DO: why error here?
		fmt.Printf("Activation ")
		cipherUtils.EvalPolyBlocks(Xint, nn.ReLUApprox.Coeffs, Box)
		for ii := range XintPlain.Blocks {
			for jj := range XintPlain.Blocks[ii] {
				utils.ActivatePlain(XintPlain.Blocks[ii][jj], nn.ReLUApprox)
			}
		}
		cipherUtils.CompareBlocks(Xint, XintPlain, Box)
		//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)

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
	fmt.Println("Corrects enc:", corrects)
	fmt.Println("Corrects clear:", correctsPlain)

	return corrects, elapsed
}

func EvalBatchEncrypted(nne *NNEnc, nnb *NNBlock, XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput, Box cipherUtils.CkksBox, labels int, debug bool) (int, time.Duration) {
	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	now := time.Now()
	for i := range nne.Weights {
		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == 0 {
			fmt.Println("Level 0, Bootstrapping...")
			cipherUtils.BootStrapBlocks(Xint, Box)
		}
		Xint, err = cipherUtils.BlocksC2CMul(Xint, nne.Weights[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
			utils.ThrowErr(err)
			fmt.Printf("Mul ")
			cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nnb.ReLUApprox.Interval), Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}

		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, nne.Bias[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(nnb.Bias[i], 1/nnb.ReLUApprox.Interval))
			utils.ThrowErr(err)
			//fmt.Printf("Bias ")
			//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(nnb.Bias[i], 1-1/nnb.ReLUApprox.Interval))
			utils.ThrowErr(err)
		}
		level = Xint.Blocks[0][0].Level()
		if level < 2 {
			//TO DO: ciphertext scale here complains
			fmt.Println("Bootstrapping for Activation")
			cipherUtils.BootStrapBlocks(Xint, Box)
		}
		//activation
		//TO DO: why error here?
		fmt.Printf("Activation ")
		cipherUtils.EvalPolyBlocks(Xint, nne.ReLUApprox.Coeffs, Box)
		if debug {
			for ii := range XintPlain.Blocks {
				for jj := range XintPlain.Blocks[ii] {
					utils.ActivatePlain(XintPlain.Blocks[ii][jj], nne.ReLUApprox)
				}
			}
			cipherUtils.CompareBlocks(Xint, XintPlain, Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
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

func EvalBatchEncryptedDistributed(nne *NNEnc, nnb *NNBlock,
	XBatchClear *plainUtils.BMatrix, Y []int, XbatchEnc *cipherUtils.EncInput,
	Box cipherUtils.CkksBox, pkQ *rlwe.PublicKey, decQ ckks.Decryptor,
	minLevel int, labels int, debug bool,
	master *distributed.DummyMaster, players []*distributed.DummyPlayer) (int, time.Duration) {

	Xint := XbatchEnc
	XintPlain := XBatchClear
	var err error
	now := time.Now()
	for i := range nne.Weights {
		fmt.Printf("======================> Layer %d\n", i+1)
		level := Xint.Blocks[0][0].Level()
		if level == minLevel { //minLevel for Bootstrapping
			fmt.Println("MinLevel, Bootstrapping...")
			for ii := 0; ii < Xint.RowP; ii++ {
				for jj := 0; jj < Xint.ColP; jj++ {
					//parallel bootstrapping of all blocks
					fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
					go func(ii, jj int) {
						Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[1], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
						utils.ThrowErr(err)
					}(ii, jj)
				}
			}
		}
		Xint, err = cipherUtils.BlocksC2CMul(Xint, nne.Weights[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.MultiPlyBlocks(XintPlain, nnb.Weights[i])
			utils.ThrowErr(err)
			fmt.Printf("Mul ")
			cipherUtils.CompareBlocks(Xint, plainUtils.MultiplyBlocksByConst(XintPlain, 1/nnb.ReLUApprox.Interval), Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}

		//bias
		Xint, err = cipherUtils.AddBlocksC2C(Xint, nne.Bias[i], Box)
		utils.ThrowErr(err)
		if debug {
			XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(nnb.Bias[i], 1/nnb.ReLUApprox.Interval))
			utils.ThrowErr(err)
			//fmt.Printf("Bias ")
			//cipherUtils.CompareBlocks(Xint, XintPlain, Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
			XintPlain, err = plainUtils.AddBlocks(XintPlain, plainUtils.MultiplyBlocksByConst(nnb.Bias[i], 1-1/nnb.ReLUApprox.Interval))
			utils.ThrowErr(err)
		}
		level = Xint.Blocks[0][0].Level()
		if level < 2 {
			fmt.Println("Level < 2 before activation , Bootstrapping...")
			for ii := 0; ii < Xint.RowP; ii++ {
				for jj := 0; jj < Xint.ColP; jj++ {
					//parallel bootstrapping of all blocks
					fmt.Println("Bootstrapping id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
					go func(ii, jj int) {
						Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[0], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
						utils.ThrowErr(err)
					}(ii, jj)
				}
			}
		}
		fmt.Printf("Activation ")
		cipherUtils.EvalPolyBlocks(Xint, nne.ReLUApprox.Coeffs, Box)
		if debug {
			for ii := range XintPlain.Blocks {
				for jj := range XintPlain.Blocks[ii] {
					utils.ActivatePlain(XintPlain.Blocks[ii][jj], nne.ReLUApprox)
				}
			}
			cipherUtils.CompareBlocks(Xint, XintPlain, Box)
			//cipherUtils.PrintDebugBlocks(Xint, XintPlain, Box)
		}
	}
	fmt.Println("Key Switch to querier secret key")
	for ii := 0; ii < Xint.RowP; ii++ {
		for jj := 0; jj < Xint.ColP; jj++ {
			//parallel key switching
			fmt.Println("Switch id:", ii*Xint.RowP+jj, " / ", Xint.RowP*Xint.ColP-1)
			go func(ii, jj int) {
				Xint.Blocks[ii][jj], err = master.InitProto(distributed.TYPES[0], pkQ, Xint.Blocks[ii][jj], ii*Xint.RowP+jj)
				utils.ThrowErr(err)
			}(ii, jj)
		}
	}
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
