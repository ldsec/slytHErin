package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"

	til "github.com/tuneinsight/tilearn"

	tim "github.com/tuneinsight/timatrices"
)

const (
	threads = 24
)

type WeightConv struct {
	W       []float64 `json:"w"`
	Rows    int       `json:"rows"`
	Cols    int       `json:"cols"`
	Kernels int       `json:"kernels"` //out
	Filters int       `json:"filters"` //in
}

type WeightDense struct {
	W    []float64 `json:"w"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

type Bias struct {
	B    []float64 `json:"b"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

type Conv struct {
	Weight WeightConv `json:"weight"`
	Bias   Bias       `json:"bias"`
}

type Dense struct {
	Weight WeightDense `json:"weight"`
	Bias   Bias        `json:"bias"`
}

type Model struct {
	C Conv `json:"conv"`
	D map[string]Dense
}

func softrelu(x float64) float64 {
	// beta = 20
	if x > -20 && x < 20 {
		return math.Log(1 + math.Exp(x))
	} else {
		if x > 0 {
			return x
		} else {
			return 0
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (math.Exp(-x) + 1)
}

func silu(x float64) float64 {
	return x * sigmoid(x)
}

func silud(x float64) float64 {
	s := sigmoid(x)
	return s * (1 + x*(1-s))
}

func reluApprox(x float64) float64 {
	interval := 20.0
	return 1.3127 + 10*x/interval + 15.7631*math.Pow(x/interval, 2) - 7.6296*math.Pow(x/interval, 4)
}

func reluApproxd(x float64) float64 {
	interval := 20.0
	return 10/interval + (2/interval)*15.7631*math.Pow(x/interval, 1) - (4/interval)*7.6296*math.Pow(x/interval, 3)
}

type SiLU struct{}

func (act *SiLU) Forward(threads int, outRaw, outActiv *tim.FloatMatrix) {
	outActiv.Func(threads, outRaw, silu)
}

func (act *SiLU) Backward(threads int, outRaw, errWeights *tim.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, silud)
}

type SoftRelu struct{}

func (act *SoftRelu) Forward(threads int, outRaw, outActiv *tim.FloatMatrix) {
	outActiv.Func(threads, outRaw, softrelu)
}

func (act *SoftRelu) Backward(threads int, outRaw, errWeights *tim.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, sigmoid)
}

type ReLUApprox struct{}

func (act *ReLUApprox) Forward(threads int, outRaw, outActiv *tim.FloatMatrix) {
	outActiv.Func(threads, outRaw, reluApprox)
}

func (act *ReLUApprox) Backward(threads int, outRaw, errWeights *tim.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, reluApproxd)
}

func idxMaxList(s []float64) (idx int) {
	var max float64
	for i, c := range s {
		if c > max {
			max = c
			idx = i
		}
	}

	return
}

func evaluateModelArgMax(model *til.Model, XTest, YTest [][]float64) {
	var TP, FP int

	dataSetTest := til.NewDataSet(XTest, nil)

	pred := model.Predict(dataSetTest)

	for j := 0; j < len(YTest); j++ {

		want := int(YTest[j][0])
		have := idxMaxList(pred[j])

		if have == want {
			TP++
		} else {
			FP++
		}
	}

	fmt.Println("Accuracy", float64(TP)/float64(TP+FP))
	fmt.Println()
}

func main() {
	rules := til.Rule{
		WithHeader:   true,
		FieldsToDrop: []string{},
		Labels:       []string{"label"},
	}

	XTrain, YTrain, err := til.CSVToSamples("mnist_train.csv", rules)

	if err != nil {
		panic(err)
	}

	XTest, YTest, err := til.CSVToSamples("mnist_test.csv", rules)

	if err != nil {
		panic(err)
	}

	validationsplit := float64(len(XTest)) / float64(len(XTrain)+len(XTest))

	for i := range XTrain {
		til.Scale(XTrain[i], 1.0/255.0)
	}

	for i := range XTest {
		til.Scale(XTest[i], 1.0/255.0)
	}

	XTrain = append(XTrain, XTest...)
	YTrain = append(YTrain, YTest...)

	imgSize := 28
	classes := 10
	batchSize := 32
	epochs := 20
	Y1HTrain := til.OneHotEncode(YTrain, classes)

	dataSetTrain := til.NewDataSet(XTrain, Y1HTrain)

	heNormal := til.NewNormalInitializer(8)
	//loss := &til.MeanSquared{}
	loss := &til.CategoricalCrossEntropy{}
	lr := 1e-6
	momentum := 0.9
	penalty := 1e-3
	optimizer := til.NewADAM(0.001, 0.9, 0.999, 1e-8)
	//optimizer := til.NewNAD(lr, momentum)
	regularizerL1L2 := &til.L1L2Regularizer{Value: penalty, L1Ratio: 0.5}
	//regularizerL1L2 := &til.VoidRegularizer{}
	activation := &ReLUApprox{}
	fmt.Println("Optimizer = Adam, Regularizer = l1l2,  Act=Relu approx")
	fmt.Printf("lr %f, momentum %f, penalty %f\n", lr, momentum, penalty)
	layerSize := 92
	layerNumber := 20 - 1

	model := til.NewModel(threads, optimizer, loss)

	filterN, filterD, pad, stride := 1, 3, 0, 1
	conv := til.NewConvolution(imgSize, imgSize, filterN, filterD, pad, stride, activation, heNormal, regularizerL1L2)
	outSize := conv.OutputSize()
	fmt.Println(outSize)
	model.Add(conv)
	model.Add(til.NewDense(outSize, layerSize, activation, heNormal, regularizerL1L2))

	//model.Add(til.NewDense(imgSize*imgSize, layerSize, &til.til.ReLU{}, heNormal))
	//model.Add(til.NewDense(layerSize, layerSize, &til.til.ReLU{}, heNormal))
	for i := 1; i < layerNumber; i++ {
		model.Add(til.NewDense(layerSize, layerSize, activation, heNormal, regularizerL1L2))
	}

	model.Add(til.NewDense(layerSize, classes, &til.Identity{}, heNormal, regularizerL1L2))
	fmt.Println(len(model.Layers))
	model.SetVerbose(1)
	model.Train(dataSetTrain, batchSize, epochs, validationsplit)

	evaluateModelArgMax(model, XTest, YTest)

	//json dump
	serialized := &Model{D: make(map[string]Dense)}
	for i, layer := range model.Layers {
		if i == 0 {
			serialized.C = Conv{}
			serialized.C.Weight = WeightConv{W: layer.Weights().M, Rows: filterD, Cols: filterD, Filters: filterN, Kernels: 1}
			serialized.C.Bias = Bias{B: layer.Bias().M, Rows: layer.Bias().Rows(), Cols: layer.Bias().Cols()}
		} else {
			key := fmt.Sprintf("dense_%d", i)
			d := Dense{}
			d.Weight = WeightDense{W: layer.Weights().M, Rows: layer.Weights().Rows(), Cols: layer.Weights().Cols()}
			d.Bias = Bias{B: layer.Bias().M, Rows: layer.Bias().Rows(), Cols: layer.Bias().Cols()}
			serialized.D[key] = d
		}
	}
	buf, err := json.Marshal(serialized)
	if err != nil {
		panic(err)
	}
	name := fmt.Sprintf("nn_%d_go.json", layerNumber+1)
	_ = ioutil.WriteFile(name, buf, 0755)
}
