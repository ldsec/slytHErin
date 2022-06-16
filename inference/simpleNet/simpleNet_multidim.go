package simpleNet

import (
	"fmt"
	md "github.com/ldsec/dnn-inference/inference/multidim"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	"gonum.org/v1/gonum/mat"
	"time"
)

type PackedMatrixLinearLayer struct {
	W  *md.PlaintextBatchMatrix //weight
	B  *md.PlaintextBatchMatrix //bias
	Mm md.MatrixMultiplication
}

type SimpleNetMD struct {
	Layers     []*PackedMatrixLinearLayer
	PmM        *md.PackedMatrixMultiplier
	Activation *utils.MinMaxPolyApprox

	innerDim, parallelBatches int
	maxR, maxC                int
}

//Converts the loaded SimpleNet model in its MultiDimentional Packing representation
//Weights and biases are assumed to be already multiplied with the constant for the activation
//Level encoding assumes the compressed model with 2 layer and 2 activations
func (sn *SimpleNet) ConvertToMDPack(parallelBatches, innerDim, inRows, inCols int, weights []*mat.Dense, biases []*mat.Dense, params ckks2.Parameters, ecd ckks2.Encoder) *SimpleNetMD {
	weightsMD := make([]*md.PackedMatrix, len(weights))
	biasesMD := make([]*md.PackedMatrix, len(biases))

	maxR := inRows
	maxC := inCols
	for i := range weights {
		//transpose to evaluate with LeftMultiplication acting as Right
		wPacked := md.PackMatrixParallelReplicated(weights[i], innerDim, parallelBatches)
		wPackedT := new(md.PackedMatrix)
		wPackedT.Transpose(wPacked)
		if wPackedT.Rows() > maxR {
			maxR = wPackedT.Rows()
		}
		if wPackedT.Cols() > maxC {
			maxC = wPackedT.Cols()
		}

		weightsMD[i] = wPackedT

		bPacked := md.PackMatrixParallelReplicated(biases[i], innerDim, parallelBatches)
		bPackedT := new(md.PackedMatrix)
		bPackedT.Transpose(bPacked)
		biasesMD[i] = bPackedT
	}
	batchEnc := md.NewBatchEncryptor(params, nil)
	snMD := new(SimpleNetMD)
	snMD.Layers = make([]*PackedMatrixLinearLayer, len(weights))
	snMD.Activation = sn.ReLUApprox

	//snMD.PmM = md.NewPackedMatrixMultiplier(box.Params, innerDim, maxR, maxC, box.Evaluator)
	level := params.MaxLevel()
	scale := params.Scale()
	for i := range weights {
		layer := new(PackedMatrixLinearLayer)
		layer.W = batchEnc.EncodeForLeftMul(level, weightsMD[i])
		mmLiteral := md.MatrixMultiplicationLiteral{
			Dimension:   innerDim,
			LevelStart:  level,
			InputScale:  params.Scale(),
			TargetScale: params.Scale(),
		}
		layer.Mm = md.NewMatrixMultiplicatonFromLiteral(params, mmLiteral, ecd)
		level -= 2 //pt mul
		layer.B = batchEnc.EncodeParallel(level, scale, biasesMD[i])
		level -= sn.ReLUApprox.LevelsOfAct() //act
	}
	snMD.maxR = maxR
	snMD.maxC = maxC
	snMD.innerDim = innerDim
	snMD.parallelBatches = parallelBatches
	return snMD
}

func (snMD *SimpleNetMD) GenerateRotations(params ckks2.Parameters) []int {
	var rotations []int
	for i := range snMD.Layers {
		rotations = append(rotations, snMD.Layers[i].Mm.Rotations(params)...)
	}
	return rotations
}

func (snMD *SimpleNetMD) InitPmMultiplier(params ckks2.Parameters, eval ckks2.Evaluator) {
	snMD.PmM = md.NewPackedMatrixMultiplier(params, snMD.innerDim, snMD.maxR, snMD.maxC, eval)
	for i := range snMD.Layers {
		snMD.PmM.AddMatrixOperation(snMD.Layers[i].Mm)
	}
}

func (snMD *SimpleNetMD) EvalBatchEncrypted(Y []int, Xenc *md.CiphertextBatchMatrix, Box md.Ckks2Box) {
	fmt.Println("Starting inference...")
	start := time.Now()

	resAfterBias := new(md.CiphertextBatchMatrix)

	for i := range snMD.Layers {
		layer := snMD.Layers[i]
		if i == 0 {
			resAfterBias = Xenc
		}
		tmpA := md.AllocateCiphertextBatchMatrix(layer.W.Rows(), resAfterBias.Cols(), snMD.innerDim, resAfterBias.M[0].Level()-2, Box.Params)
		snMD.PmM.MulPlainLeft([]*md.PlaintextBatchMatrix{layer.W}, resAfterBias, snMD.innerDim, []*md.CiphertextBatchMatrix{tmpA})
		tmpB := md.AllocateCiphertextBatchMatrix(tmpA.Rows(), tmpA.Cols(), snMD.innerDim, tmpA.M[0].Level(), Box.Params)
		snMD.PmM.AddPlain(tmpA, layer.B, tmpB)
		snMD.PmM.EvalPoly(tmpB, ckks2.NewPoly(snMD.Activation.Poly.Coeffs))
		resAfterBias = tmpB
	}
	elapsed := time.Since(start)
	fmt.Println("Done: ", elapsed)

	resCipher := md.UnpackCipherParallel(resAfterBias, snMD.innerDim, 10, len(Y), Box.Encoder, Box.Decryptor, Box.Params, snMD.parallelBatches)
	resCipher2 := plainUtils.TransposeDense(mat.NewDense(len(Y), 10, resCipher))
	predictions := make([]int, len(Y))
	corrects := 0
	for i := 0; i < len(Y); i++ {
		maxIdx := 0
		maxConfidence := 0.0
		for j := 0; j < len(Y); j++ {
			confidence := resCipher2.At(i, j)
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
	fmt.Println("Corrects enc:", corrects)
}
