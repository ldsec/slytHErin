/*
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

	return math.Log(1 + math.Exp(x))

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
	epochs := 30
	Y1HTrain := til.OneHotEncode(YTrain, classes)

	dataSetTrain := til.NewDataSet(XTrain, Y1HTrain)

	heNormal := til.NewNormalInitializer(8)
	loss := &til.MeanSquared{}
	//loss := &til.CategoricalCrossEntropy{}
	optimizer := til.NewADAM(3e-5, 0.9, 0.999, 1e-8)
	regularizerL1L2 := &til.L1L2Regularizer{Value: 1e-6, L1Ratio: 0.5}

	layerSize := 92
	layerNumber := 20 - 1

	model := til.NewModel(threads, til.Shape{imgSize, imgSize, 1}, optimizer, loss)

	filterN, filterD, pad, stride := 1, 3, 0, 1
	model.Add(til.NewConvolution(filterN, filterD, pad, stride, &SoftRelu{}, regularizerL1L2, heNormal))

	model.Add(til.NewDense(layerSize, &SoftRelu{}, regularizerL1L2, heNormal))

	for i := 1; i < layerNumber; i++ {
		model.Add(til.NewDense(layerSize, &SoftRelu{}, regularizerL1L2, heNormal))
	}

	model.Add(til.NewDense(classes, &til.Identity{}, regularizerL1L2, heNormal))

	model.SetVerbose(1)
	model.Train(dataSetTrain, batchSize, epochs, validationsplit)

	evaluateModelArgMax(model, XTest, YTest)

	//json dump
	serialized := &Model{D: make(map[string]Dense)}
	for i, l := range model.Layers {
		if i == 0 {
			layer, _ := l.(*til.Convolution)
			serialized.C = Conv{}
			serialized.C.Weight = WeightConv{W: layer.Weights()[0].M, Rows: filterD, Cols: filterD, Filters: filterN, Kernels: 1}
			serialized.C.Bias = Bias{B: layer.Bias().M, Rows: layer.Bias().Rows(), Cols: layer.Bias().Cols()}
		} else {
			key := fmt.Sprintf("dense_%d", i)
			layer := l.(*til.Dense)
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
*/

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"

	mat "github.com/tuneinsight/mat"
	til "github.com/tuneinsight/tilearn"
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

var a = -80.0
var b = 80.0
var coeffsF = []float64{25.45711877649942, 40.0, 16.991850131857497, -2.7068294695622863e-15, -3.4105646304321975, -8.571626653613907e-15, 1.4702852988286828, -1.3534147347811431e-14, -0.8234165806973575, -1.98500827767901e-14, 0.5292705427041939, -4.5564962737631824e-14, -0.37076815389704754, -6.767073673905716e-14, 0.27555263533356034, -8.300943706657678e-14, -0.21383292294720038, -8.932537249555545e-14, 0.1715003018260692, -6.767073673905716e-14, -0.14116722325469067, -4.240699502314249e-14, 0.1186607009215158, -8.571626653613907e-15, -0.10147935947413536, 9.473903143468002e-15, 0.08805049482576115, 1.3534147347811431e-14, -0.07734576651584678, 1.6015407694910195e-14, 0.06867042818832568, 3.0000693287648675e-14, -0.06154196299437711, 5.3234312901391633e-14, 0.05561725935172035, 2.210577400142534e-14, -0.05064734246447628, 1.1955163490566765e-14, 0.046448361995931324, 3.0000693287648675e-14, -0.042882496810636085, 2.0301221021717147e-15, 0.03984509101025014, -4.624167010502239e-14, -0.03725580958584757, -1.001526903738046e-13, 0.03505244893764464, -1.9804968952297396e-13, -0.033186538899772904, -2.7102130063992395e-13, 0.03162017785039036, -2.8218697220186837e-13, -0.030323732620106382, -2.987663027029374e-13, 0.029274156137741137, -2.8167944167632545e-13, -0.028453754869487126, -1.7893270639419032e-13, 0.02784929073330891, -3.310226872152213e-14, -0.02745133818777235, 3.1255421531352025e-14, 0.027253842345891288}
var coeffsG = []float64{0.5000000000000001, 0.6366717916704755, -2.044220172325685e-16, -0.21236017729654227, 3.52451753849256e-17, 0.1275717400406127, -2.678633329254346e-16, -0.0912755379713117, -5.639228061588096e-17, 0.07113079024083607, -4.65236315081018e-16, -0.05831428234255675, -6.908054375445419e-16, 0.049430227227460176, -9.586687704699764e-16, -0.04289227822601032, -1.1313701298561118e-15, 0.03785943939889505, -1.050306226470783e-15, -0.03384404327630899, -6.908054375445419e-16, 0.03054374143700403, -2.925349556948825e-16, -0.027761144179757623, 1.0926004369326937e-16, 0.025362019319006925, 6.344131569286608e-17, -0.023252051630116764, 3.841724116956891e-16, 0.02136319929695002, -6.344131569286608e-17, -0.019645311251574454, 8.035899987763038e-16, 0.018060772677054197, 5.780208763127799e-16, -0.016580967488936796, -1.2335811384723962e-16, 0.015183871057587688, 4.6876083261951055e-16, -0.013852368681154241, 2.9605947323337506e-16, 0.012573053504094297, -2.678633329254346e-16, -0.011335349402383009, -8.811293846231401e-16, 0.010130859323981665, -1.540214164321249e-15, -0.008952873377305472, -3.471649775415172e-15, 0.007795992251798637, -3.2672277581826034e-15, -0.006655835252381858, -3.704267932955681e-15, 0.005528811208096236, -3.8875428449572946e-15, -0.004411936498779423, -3.061043482180789e-15, 0.003302688483896145, -1.4644370372436588e-15, -0.0021988853745102442, 6.167905692361981e-16, 0.001098585474657349, 2.1675782861729246e-16}

func evaluatePoly(coeffs []float64, x, a, b float64) (y float64) {
	Tprev := 1.0
	x = (2*x - a - b) / (b - a)
	T := x
	y = coeffs[0]
	for i := 1; i < len(coeffs); i++ {
		y += coeffs[i] * T
		Tnext := 2*x*T - Tprev
		Tprev = T
		T = Tnext
	}
	return
}

func sigmoid(x float64) float64 {
	return 1 / (math.Exp(-x) + 1)
}

func silu(x float64) float64 {
	return evaluatePoly(coeffsF, x, a, b)
	//return x * sigmoid(x)
}

func silud(x float64) float64 {
	//s := sigmoid(x)
	//return s * (1 + x*(1-s))
	return evaluatePoly(coeffsG, x, a, b)
}

type SiLU struct{}

func (act *SiLU) Forward(threads int, outRaw, outActiv *mat.FloatMatrix) {
	outActiv.Func(threads, outRaw, silu)
}

func (act *SiLU) Backward(threads int, outRaw, errWeights *mat.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, silud)
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
	Y1HTrain := til.OneHotEncode(YTrain, classes)

	dataSetTrain := til.NewDataSet(XTrain, Y1HTrain)

	heNormal := til.NewNormalInitializer(8)
	loss := &til.MeanSquared{}
	//optimizer := til.NewADAM(0.0001, 0.9, 0.999, 1e-8)
	optimizer := til.NewSGD(0.01, 0.001)
	//regularizerL1L2 := &til.L1L2Regularizer{Value: 1e-7, L1Ratio: 0.5}
	regularizerL1L2 := &til.VoidRegularizer{}

	layerSize := 92
	layerNumber := 50

	epochs := 15

	if layerNumber == 50 {
		optimizer = til.NewSGD(0.0001, 0.0001)
		epochs = 100
	}

	model := til.NewModel(threads, til.Shape{Height: imgSize, Width: imgSize, Channel: 1}, optimizer, loss)

	filterN, filterD, pad, stride := 1, 3, 0, 1
	model.Add(til.NewConvolution(filterN, filterD, pad, stride, &SiLU{}, regularizerL1L2, heNormal))

	for i := 1; i < layerNumber; i++ {
		model.Add(til.NewDense(layerSize, &SiLU{}, heNormal, regularizerL1L2))
	}

	model.Add(til.NewDense(classes, &til.SoftMax{}, heNormal, regularizerL1L2))

	model.SetVerbose(1)
	//model.SetHistory(true)
	model.Train(dataSetTrain, batchSize, epochs, validationsplit)

	//for _, layer := range model.Layers {
	//	fmt.Println(layer.(*til.Dense).History)
	//	fmt.Println()
	//}

	evaluateModelArgMax(model, XTest, YTest)
	//json dump
	serialized := &Model{D: make(map[string]Dense)}
	for i, l := range model.Layers {
		if i == 0 {
			layer, _ := l.(*til.Convolution)
			serialized.C = Conv{}
			serialized.C.Weight = WeightConv{W: layer.Weights()[0].M, Rows: filterD, Cols: filterD, Filters: filterN, Kernels: 1}
			serialized.C.Bias = Bias{B: layer.Bias().M, Rows: layer.Bias().Rows(), Cols: layer.Bias().Cols()}
		} else {
			key := fmt.Sprintf("dense_%d", i)
			layer := l.(*til.Dense)
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
	name := fmt.Sprintf("nn%d_poly_go.json", layerNumber)
	_ = ioutil.WriteFile(name, buf, 0755)
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
