package nn

import (
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"io/ioutil"
	"os"
	"time"
)

type NN struct {
	Conv       utils.Layer      `json:"conv"`
	Dense      []utils.Layer    `json:"dense"`
	Layers     int              `json:"layers"`
	ReLUApprox utils.PolyApprox //this will store the coefficients of the poly approximating ReLU

	RowsOutConv, ColsOutConv, ChansOutConv, DimOutDense int //dimentions
}

func LoadNN(path string) *NN {
	// loads json file with weights
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

func (nn *NN) Init() {
	//init activation
	nn.ReLUApprox = utils.InitReLU()

	//init dimentional values
	nn.RowsOutConv = 21
	nn.ColsOutConv = 20
	nn.ChansOutConv = 2 //tot is 21*20*2 = 840 per sample after conv
	nn.DimOutDense = 92 //per sample, all dense are the same but last one which is 92x10
}

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
