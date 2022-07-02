package cryptonet

import (
	"fmt"
	md "github.com/ldsec/dnn-inference/inference/multidim"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/ldsec/dnn-inference/inference/utils"
	ckks2 "github.com/ldsec/lattigo/v2/ckks"
	rlwe2 "github.com/ldsec/lattigo/v2/rlwe"
	"gonum.org/v1/gonum/mat"
	"time"
)

type PackedMatrixLinearLayer struct {
	W  *md.PlaintextBatchMatrix //weight
	B  *md.PlaintextBatchMatrix //bias
	Mm md.MatrixMultiplication
}

type cryptonetMD struct {
	Layers     []*PackedMatrixLinearLayer
	PmM        *md.PackedMatrixMultiplier
	Activation *utils.MinMaxPolyApprox

	innerDim, parallelBatches int
	maxR, maxC                int
}

//HELPERS
//takes the batch as dense matrix and packs it (transpose)
func PackBatchParallel(X *mat.Dense, innerDim int, params ckks2.Parameters) *md.PackedMatrix {
	//Xpacked := md.PackMatrixParallel(X, innerDim, params.LogSlots())
	Xpacked := md.PackMatrixParallel(X, innerDim, params.LogSlots())
	XpackedT := new(md.PackedMatrix)
	XpackedT.Transpose(Xpacked)
	return XpackedT
}

//Converts the loaded cryptonet model in its MultiDimentional Packing representation
//Weights and biases are assumed to be already multiplied with the constant for the activation
//Level encoding assumes the compressed model with 2 layer and 2 activations
func (sn *cryptonet) ConvertToMDPack(parallelBatches, innerDim, inRows, inCols int, weights []*mat.Dense, biases []*mat.Dense, params ckks2.Parameters, ecd ckks2.Encoder) *cryptonetMD {
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

		bPacked := md.PackMatrixParallel(biases[i], innerDim, params.LogSlots())
		bPackedT := new(md.PackedMatrix)
		bPackedT.Transpose(bPacked)
		biasesMD[i] = bPackedT
		fmt.Printf("Layer %d:\n", i+1)
		fmt.Printf("Weight Dim (Rows,Cols): %d, %d \n", wPackedT.Rows(), wPackedT.Cols())
		fmt.Printf("Bias Dim (Rows,Cols): %d, %d \n", bPackedT.Rows(), bPackedT.Cols())
	}
	batchEnc := md.NewBatchEncryptor(params, rlwe2.NewPublicKey(params.Parameters)) //it shouldn't be used
	snMD := new(cryptonetMD)
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
		snMD.Layers[i] = layer
	}
	snMD.maxR = maxR
	snMD.maxC = maxC
	snMD.innerDim = innerDim
	snMD.parallelBatches = parallelBatches
	return snMD
}

func (snMD *cryptonetMD) GenerateRotations(params ckks2.Parameters) []int {
	var rotations []int
	for i := range snMD.Layers {
		rotations = append(rotations, snMD.Layers[i].Mm.Rotations(params)...)
	}
	return rotations
}

func (snMD *cryptonetMD) InitPmMultiplier(params ckks2.Parameters, eval ckks2.Evaluator) {
	snMD.PmM = md.NewPackedMatrixMultiplier(params, snMD.innerDim, snMD.maxR, snMD.maxC, eval)
	for i := range snMD.Layers {
		snMD.PmM.AddMatrixOperation(snMD.Layers[i].Mm)
	}
}

func (snMD *cryptonetMD) EvalBatchEncrypted(Y []int, Xenc *md.CiphertextBatchMatrix, Box md.Ckks2Box) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	resAfterBias := new(md.CiphertextBatchMatrix)
	fmt.Println("Start level: ", Xenc.Level())
	for i := range snMD.Layers {
		layer := snMD.Layers[i]
		if i == 0 {
			resAfterBias = Xenc
		}
		fmt.Println("Layer ", i+1)
		tmpA := md.AllocateCiphertextBatchMatrix(layer.W.Rows(), resAfterBias.Cols(), snMD.innerDim, resAfterBias.M[0].Level(), Box.Params)
		snMD.PmM.MulPlainLeft([]*md.PlaintextBatchMatrix{layer.W}, resAfterBias, snMD.innerDim, []*md.CiphertextBatchMatrix{tmpA})
		fmt.Println("Level after Mul: ", tmpA.Level())

		tmpB := md.AllocateCiphertextBatchMatrix(tmpA.Rows(), tmpA.Cols(), snMD.innerDim, tmpA.M[0].Level(), Box.Params)
		snMD.PmM.AddPlain(tmpA, layer.B, tmpB)
		fmt.Println("Level after Add: ", tmpB.Level())

		if i != len(snMD.Layers)-1 {
			resAfterBias = snMD.PmM.EvalPoly(tmpB, ckks2.NewPoly(snMD.Activation.Poly.Coeffs))
			fmt.Println("Level after Act: ", tmpB.Level())
		}
	}
	elapsed := time.Since(start)
	fmt.Println("Done: ", elapsed)

	//tranpose the result
	resCipher := md.DecryptCipher(resAfterBias, Box.Encoder, Box.Decryptor, Box.Params, snMD.parallelBatches)
	resCipherT := new(md.PackedMatrix)
	resCipherT.Transpose(resCipher)
	resCipher2 := mat.NewDense(len(Y), 10, md.UnpackMatrixParallel(resCipherT, snMD.innerDim, len(Y), 10))
	corrects, accuracy, _ := utils.Predict(Y, 10, plainUtils.MatToArray(resCipher2))
	fmt.Println("Corrects enc:", corrects)
	return utils.Stats{
		Corrects: corrects,
		Time:     elapsed,
		Accuracy: accuracy,
	}
}

func (snMD *cryptonetMD) EvalBatchEncrypted_Debug(Y []int, Xenc *md.CiphertextBatchMatrix, Box md.Ckks2Box, Xclear *mat.Dense, weights, biases []*mat.Dense) utils.Stats {
	fmt.Println("Starting inference...")
	start := time.Now()

	resAfterBias := new(md.CiphertextBatchMatrix)
	resAfterBiasClear := new(mat.Dense)
	fmt.Println("Start level: ", Xenc.Level())
	for i := range snMD.Layers {
		layer := snMD.Layers[i]
		if i == 0 {
			resAfterBias = Xenc
			resAfterBiasClear = Xclear
		}
		fmt.Println("Layer ", i+1)
		tmpA := md.AllocateCiphertextBatchMatrix(layer.W.Rows(), resAfterBias.Cols(), snMD.innerDim, resAfterBias.M[0].Level(), Box.Params)
		snMD.PmM.MulPlainLeft([]*md.PlaintextBatchMatrix{layer.W}, resAfterBias, snMD.innerDim, []*md.CiphertextBatchMatrix{tmpA})

		fmt.Println("Level after Mul: ", tmpA.Level())
		tmpAclear := new(mat.Dense)
		tmpAclear.Mul(resAfterBiasClear, weights[i])
		utils.ThrowErr(md.DebugMD(plainUtils.MulByConst(tmpAclear, 1.0/snMD.Activation.Interval), tmpA, snMD.innerDim, snMD.parallelBatches, true, false, Box))

		tmpB := md.AllocateCiphertextBatchMatrix(tmpA.Rows(), tmpA.Cols(), snMD.innerDim, tmpA.M[0].Level(), Box.Params)
		snMD.PmM.AddPlain(tmpA, layer.B, tmpB)

		tmpBclear := new(mat.Dense)
		fmt.Println("Level after Add: ", tmpB.Level())
		tmpBclear.Add(tmpAclear, biases[i])
		utils.ThrowErr(md.DebugMD(plainUtils.MulByConst(tmpBclear, 1.0/snMD.Activation.Interval), tmpB, snMD.innerDim, snMD.parallelBatches, true, false, Box))

		resAfterBias = snMD.PmM.EvalPoly(tmpB, ckks2.NewPoly(snMD.Activation.Poly.Coeffs))
		utils.ActivatePlain(tmpBclear, snMD.Activation)
		fmt.Println("Level after Act: ", tmpB.Level())
		utils.ThrowErr(md.DebugMD(tmpBclear, resAfterBias, snMD.innerDim, snMD.parallelBatches, true, false, Box))

		resAfterBiasClear = tmpBclear
	}
	elapsed := time.Since(start)
	fmt.Println("Done: ", elapsed)

	//tranpose the result
	resCipher := md.DecryptCipher(resAfterBias, Box.Encoder, Box.Decryptor, Box.Params, snMD.parallelBatches)
	resCipherT := new(md.PackedMatrix)
	resCipherT.Transpose(resCipher)
	resCipher2 := mat.NewDense(len(Y), 10, md.UnpackMatrixParallel(resCipherT, snMD.innerDim, len(Y), 10))

	corrects, accuracy, _ := utils.Predict(Y, 10, plainUtils.MatToArray(resCipher2))
	fmt.Println("Corrects enc:", corrects)
	return utils.Stats{
		Corrects: corrects,
		Time:     elapsed,
		Accuracy: accuracy,
	}
}
