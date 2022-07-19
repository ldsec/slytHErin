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

//deg 63
var a = -80.0
var b = 80.0

//F and G for silu and d
var coeffsF = []float64{25.45711877649942, 40.0, 16.991850131857497, -2.7068294695622863e-15, -3.4105646304321975, -8.571626653613907e-15, 1.4702852988286828, -1.3534147347811431e-14, -0.8234165806973575, -1.98500827767901e-14, 0.5292705427041939, -4.5564962737631824e-14, -0.37076815389704754, -6.767073673905716e-14, 0.27555263533356034, -8.300943706657678e-14, -0.21383292294720038, -8.932537249555545e-14, 0.1715003018260692, -6.767073673905716e-14, -0.14116722325469067, -4.240699502314249e-14, 0.1186607009215158, -8.571626653613907e-15, -0.10147935947413536, 9.473903143468002e-15, 0.08805049482576115, 1.3534147347811431e-14, -0.07734576651584678, 1.6015407694910195e-14, 0.06867042818832568, 3.0000693287648675e-14, -0.06154196299437711, 5.3234312901391633e-14, 0.05561725935172035, 2.210577400142534e-14, -0.05064734246447628, 1.1955163490566765e-14, 0.046448361995931324, 3.0000693287648675e-14, -0.042882496810636085, 2.0301221021717147e-15, 0.03984509101025014, -4.624167010502239e-14, -0.03725580958584757, -1.001526903738046e-13, 0.03505244893764464, -1.9804968952297396e-13, -0.033186538899772904, -2.7102130063992395e-13, 0.03162017785039036, -2.8218697220186837e-13, -0.030323732620106382, -2.987663027029374e-13, 0.029274156137741137, -2.8167944167632545e-13, -0.028453754869487126, -1.7893270639419032e-13, 0.02784929073330891, -3.310226872152213e-14, -0.02745133818777235, 3.1255421531352025e-14, 0.027253842345891288}
var coeffsG = []float64{0.5000000000000001, 0.6366717916704755, -2.044220172325685e-16, -0.21236017729654227, 3.52451753849256e-17, 0.1275717400406127, -2.678633329254346e-16, -0.0912755379713117, -5.639228061588096e-17, 0.07113079024083607, -4.65236315081018e-16, -0.05831428234255675, -6.908054375445419e-16, 0.049430227227460176, -9.586687704699764e-16, -0.04289227822601032, -1.1313701298561118e-15, 0.03785943939889505, -1.050306226470783e-15, -0.03384404327630899, -6.908054375445419e-16, 0.03054374143700403, -2.925349556948825e-16, -0.027761144179757623, 1.0926004369326937e-16, 0.025362019319006925, 6.344131569286608e-17, -0.023252051630116764, 3.841724116956891e-16, 0.02136319929695002, -6.344131569286608e-17, -0.019645311251574454, 8.035899987763038e-16, 0.018060772677054197, 5.780208763127799e-16, -0.016580967488936796, -1.2335811384723962e-16, 0.015183871057587688, 4.6876083261951055e-16, -0.013852368681154241, 2.9605947323337506e-16, 0.012573053504094297, -2.678633329254346e-16, -0.011335349402383009, -8.811293846231401e-16, 0.010130859323981665, -1.540214164321249e-15, -0.008952873377305472, -3.471649775415172e-15, 0.007795992251798637, -3.2672277581826034e-15, -0.006655835252381858, -3.704267932955681e-15, 0.005528811208096236, -3.8875428449572946e-15, -0.004411936498779423, -3.061043482180789e-15, 0.003302688483896145, -1.4644370372436588e-15, -0.0021988853745102442, 6.167905692361981e-16, 0.001098585474657349, 2.1675782861729246e-16}

//H and K for sigmoid and d
var coeffsH = []float64{5.00000000e-01, 6.36456060e-01, 9.53205768e-15, -2.11716162e-01, 9.48976347e-15, 1.26508924e-01, 9.47830879e-15, -8.98095414e-02, 9.42789463e-15, 6.92831071e-02, 9.43264595e-15, -5.61119185e-02, 9.47742766e-15, 4.69052589e-02, 9.50493246e-15, -4.00813435e-02, 9.46581709e-15, 3.48031789e-02, 9.44570700e-15, -3.05865070e-02, 9.50214209e-15, 2.71317784e-02, 9.45058169e-15, -2.42437913e-02, 9.53685617e-15, 2.17899042e-02, 9.47327879e-15, -1.96768221e-02, 9.39623685e-15, 1.78369915e-02, 9.49744607e-15, -1.62202632e-02, 9.50851682e-15, 1.47885917e-02, 9.45813211e-15, -1.35125538e-02, 9.44570700e-15, 1.23689999e-02, 9.39763353e-15, -1.13394315e-02, 9.47111422e-15, 1.04088561e-02, 9.48464947e-15, -9.56496439e-03, 9.51826572e-15, 8.79752982e-03, 9.51089521e-15, -8.09796317e-03, 9.40769239e-15, 7.45897833e-03, 9.47408061e-15, -6.87433877e-03, 9.49233786e-15, 6.33866336e-03, 9.44570700e-15, -5.84727375e-03, 9.46591186e-15, 5.39607989e-03, 9.53675876e-15, -4.98148456e-03, 9.38681926e-15, 4.60031215e-03, 9.93341184e-15, -2.90492578e-02}
var coeffsK = []float64{-1.71834366e-02, 2.35520903e-13, -5.02782746e-02, 2.35044300e-13, -3.43995624e-02, 2.34095324e-13, -5.02131779e-02, 2.32673577e-13, -3.44965081e-02, 2.30787999e-13, -5.00852072e-02, 2.28429837e-13, -3.46544296e-02, 2.25586609e-13, -4.98986388e-02, 2.22259882e-13, -3.48681350e-02, 2.18473556e-13, -4.96594860e-02, 2.14222987e-13, -3.51308952e-02, 2.09471916e-13, -4.93750789e-02, 2.04274096e-13, -3.54348989e-02, 1.98551983e-13, -4.90535890e-02, 1.92394352e-13, -3.57717341e-02, 1.85816986e-13, -4.87035529e-02, 1.78693901e-13, -3.61328489e-02, 1.71087088e-13, -4.83334371e-02, 1.63047675e-13, -3.65099525e-02, 1.54546539e-13, -4.79512775e-02, 1.45618787e-13, -3.68953318e-02, 1.36147673e-13, -4.75644092e-02, 1.26188791e-13, -3.72820725e-02, 1.15718699e-13, -4.71792936e-02, 1.04781169e-13, -3.76641868e-02, 9.34919385e-14, -4.68014353e-02, 8.16493377e-14, -3.80366534e-02, 6.93092985e-14, -4.64353823e-02, 5.65575940e-14, -3.83953809e-02, 4.33053174e-14, -4.60847948e-02, 2.94770172e-14, -3.87371050e-02, 1.53967883e-14}

//var a = -100.0
//var b = 100.0
//var coeffsF = []float64{3.18257500e+01, 5.00000000e+01, 2.12311220e+01, 6.75373690e-12, -4.25455145e+00, -9.32277183e-13, 1.82926160e+00, -7.02076253e-12, -1.02075605e+00, 7.71385316e-12, 6.53172249e-01, 2.06113786e-14, -4.55158163e-01, -7.45241498e-12, 3.36263994e-01, 5.92814442e-12, -2.59246611e-01, 2.87194607e-12, 2.06465427e-01, -8.53282657e-12, -1.68681015e-01, 3.81062995e-12, 1.40671937e-01, 6.60066631e-12, -1.19306141e-01, -1.09006221e-11, 1.02612354e-01, 3.63975867e-12, -8.92997696e-02, 7.71946440e-12, 7.84946298e-02, -1.06516881e-11, -6.95885620e-02, 1.68515070e-12, 6.21475683e-02, 9.20682930e-12, -5.58554553e-02, -9.53730473e-12, 5.04775607e-02, -1.71862961e-12, -4.58368720e-02, 1.23989969e-11, 4.17979186e-02, -1.03536731e-11, -3.82556758e-02, -2.82915269e-12, 3.51277827e-02, 1.32440373e-11, -3.23489850e-02, -9.60144959e-12, 2.98671096e-02, -4.14225633e-12, -2.76401201e-02, 1.28132284e-11, 2.56339163e-02, -6.77623686e-12, -2.38207088e-02, -7.50767766e-12, 2.21777772e-02, 1.43888956e-11, -2.06865425e-02, 1.46055307e-12, 1.54706216e-01, -8.43143024e-12}
//var coeffsG = []float64{5.00000000e-01, 8.16367742e-01, -2.80830698e-13, -3.28771359e-02, -6.86054912e-13, 3.07486980e-01, -5.92827194e-13, 8.79755879e-02, 3.90079560e-13, 2.51296555e-01, -9.98414009e-13, 1.20662105e-01, -1.00294851e-12, 2.29900065e-01, 9.34679382e-13, 1.35746146e-01, -8.43763945e-13, 2.18705062e-01, -1.82022561e-12, 1.44377508e-01, 1.42224849e-12, 2.11849914e-01, -1.78216091e-13, 1.49954262e-01, -3.21452260e-12, 2.07221209e-01, 2.23578846e-12, 1.53862785e-01, 2.70318780e-13, 2.03870656e-01, -4.20697057e-12, 1.56773878e-01, 2.39707602e-12, 2.01310558e-01, 1.28487656e-12, 1.59050211e-01, -5.15990395e-12, 1.99266139e-01, 1.89770154e-12, 1.60903193e-01, 3.23823264e-12, 1.97572691e-01, -6.92894480e-12, 1.62462439e-01, 1.97521408e-12, 1.96127434e-01, 4.52145150e-12, 1.63809874e-01, -7.92794356e-12, 1.94864899e-01, 1.48147704e-12, 1.64997790e-01, 5.70657849e-12, 1.93743515e-01, -7.87544364e-12, 1.66058885e-01, -4.21583095e-13, 1.92738079e-01, 8.13716943e-12, 1.67011857e-01, -8.84172737e-12, 1.91835708e-01}

func evaluatePoly(coeffs []float64, x, a, b float64) (y float64) {
	Tprev := 1.0
	x = (2*x - a - b) / (b - a)
	T := x
	y = coeffs[0] * 1
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

func siluApprox(k int, x float64) float64 {
	return evaluatePoly(coeffsF, x, a, b)
	//return x * sigmoid(x)
}

func siludApprox(k int, x float64) float64 {
	//s := sigmoid(x)
	//return s * (1 + x*(1-s))
	return evaluatePoly(coeffsG, x, a, b)
}

func sigApprox(k int, x float64) float64 {
	return evaluatePoly(coeffsH, x, a, b)
	//return x * sigmoid(x)
}

func sigdApprox(k int, x float64) float64 {
	//s := sigmoid(x)
	//return s * (1 + x*(1-s))
	return evaluatePoly(coeffsK, x, a, b)
}

func silu(k int, x float64) float64 {
	return x * sigmoid(x)
}

func silud(k int, x float64) float64 {
	s := sigmoid(x)
	return s * (1 + x*(1-s))

}

type SiLUApprox struct {
	til.Activation
}

func (act *SiLUApprox) Forward(threads, epoch int, outRaw, outActiv *mat.FloatMatrix) {
	outActiv.Func(threads, outRaw, siluApprox)
}

func (act *SiLUApprox) Backward(threads, epoch int, outRaw, errWeights *mat.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, siludApprox)
}

type SigmoidApprox struct {
	til.Activation
}

func (act *SigmoidApprox) Forward(threads, epoch int, outRaw, outActiv *mat.FloatMatrix) {
	outActiv.Func(threads, outRaw, sigApprox)
}

func (act *SigmoidApprox) Backward(threads, epoch int, outRaw, errWeights *mat.FloatMatrix) {
	errWeights.FuncAndDot(threads, outRaw, sigdApprox)
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
		optimizer = til.NewSGD(0.0005, 1e-5)
		epochs = 30
		batchSize = 32
	}

	model := til.NewModel(threads, til.Shape{Height: imgSize, Width: imgSize, Channel: 1}, optimizer, loss)

	filterN, filterD, pad, stride := 1, 3, 0, 1
	model.Add(til.NewConvolution(filterN, filterD, pad, stride, &SiLUApprox{}, regularizerL1L2, heNormal))

	for i := 1; i < layerNumber; i++ {
		if i+1%5 != 0 {
			model.Add(til.NewDense(layerSize, &SiLUApprox{}, heNormal, regularizerL1L2))
		} else if layerNumber == 50 {
			//NN50 -> every ~ 5 layers add sigmoid
			model.Add(til.NewDense(layerSize, &SigmoidApprox{}, heNormal, regularizerL1L2))
		}
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
