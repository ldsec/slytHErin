package modelsEnc

import (
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/cipherUtils"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math"
	"os"
	"time"
)

type Bias struct {
	B   []float64 `json:"b"`
	Len int       `json:"len"`
}

type Kernel struct {
	W    []float64 `json:"w"` //Matrix M s.t X @ M = conv(X, layer).flatten() where X is a row-flattened data sample (this actually can represent also a regular dense layer)
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

type Layer struct {
	Weight Kernel `json:"weight"`
	Bias   Bias   `json:"bias"`
}
type PolyApprox struct {
	Interval float64
	Degree   int
	Coeffs   []float64
}
type NN struct {
	Conv       Layer      `json:"conv"`
	Dense      []Layer    `json:"dense"`
	Layers     int        `json:"layers"`
	ReLUApprox PolyApprox //this will store the coefficients of the poly approximating ReLU

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
	nn.ReLUApprox.Degree = 3
	nn.ReLUApprox.Interval = 10.0
	nn.ReLUApprox.Coeffs = make([]float64, nn.ReLUApprox.Degree)
	nn.ReLUApprox.Coeffs[0] = 1.1155
	nn.ReLUApprox.Coeffs[1] = 5
	nn.ReLUApprox.Coeffs[2] = 4.4003

	//init dimentional values
	nn.RowsOutConv = 21
	nn.ColsOutConv = 20
	nn.ChansOutConv = 2 //tot is 21*20*2 = 840 per sample after conv
	nn.DimOutDense = 92 //per sample, all dense are the same but last one which is 92x10
}

func buildKernelMatrix(k Kernel) *mat.Dense {
	/*
		Returns a matrix M s.t X.M = conv(x,layer)
	*/

	res := mat.NewDense(k.Rows, k.Cols, nil)
	for i := 0; i < k.Rows; i++ {
		for j := 0; j < k.Cols; j++ {
			res.Set(i, j, k.W[i*k.Cols+j])
		}
	}
	return res
}

func buildBiasMatrix(b Bias, cols, batchSize int) *mat.Dense {
	// Compute a matrix containing the bias of the layer, to be added to the result
	res := mat.NewDense(batchSize, cols, nil)
	for i := 0; i < batchSize; i++ {
		res.SetRow(i, plainUtils.Pad(b.B, cols-len(b.B)))
	}
	return res
}

func (nn *NN) ActivatePlain(X *mat.Dense) {
	/*
		Apply the activation function elementwise
	*/
	rows, cols := X.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := X.At(r, c) / float64(nn.ReLUApprox.Interval)
			res := 0.0
			for deg := 0; deg < nn.ReLUApprox.Degree; deg++ {
				res += (math.Pow(v, float64(deg)) * nn.ReLUApprox.Coeffs[deg])
			}
			X.Set(r, c, res)
		}
	}
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
		cipherUtils.CompareBlocks(Xint, XintPlain, Box)
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
				nn.ActivatePlain(XintPlain.Blocks[ii][jj])
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
		predictions[i] = maxIdx
		if predictions[i] == Y[i] {
			correctsPlain += 1
		}
	}
	fmt.Println("Corrects enc:", corrects)
	fmt.Println("Corrects clear:", correctsPlain)

	return corrects, elapsed
}
