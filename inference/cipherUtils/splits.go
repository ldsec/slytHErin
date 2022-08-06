package cipherUtils

import (
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"github.com/tuneinsight/lattigo/v3/utils"
	"math"
)

type BlockSplits struct {
	InnerRows, InnerCols, RowP, ColP int
}

//Information on the current split
type SplitsInfo struct {
	InputRows     int   `json:"input_rows,omitempty"`
	InputCols     int   `json:"input_cols,omitempty"` //inner dims of input
	InputRowP     int   `json:"input_row_p,omitempty"`
	InputColP     int   `json:"input_col_p,omitempty"` //partition of input
	NumWeights    int   `json:"num_weights,omitempty"`
	RowsOfWeights []int `json:"rows_of_weights,omitempty"` //inner rows of weights
	ColsOfWeights []int `json:"cols_of_weights,omitempty"`
	RowPOfWeights []int `json:"row_p_of_weights,omitempty"` //row partition of weights
	ColPOfWeights []int `json:"col_p_of_weights,omitempty"`
}

func GetFillRatio(rows, cols, replicaFactor int, slotsAvailable float64) float64 {
	consumedSlots := float64(rows * cols * replicaFactor)
	return consumedSlots / slotsAvailable
}

//Computes the optimal number of rows for input sub-matrices. Takes the innerCols of the Input and the maxInnerCols of all the weights in the pipeline
func GetOptimalInnerRows(inputInnerCols int, maxInnerCols int, params ckks.Parameters) int {
	innerCols := plainUtils.Max(inputInnerCols, maxInnerCols)
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots()-1)))
	optInnerRows := int(math.Floor(slotsAvailable / float64(innerCols)))
	//takes into account that if maxInnerCols > inputInnerRows we will have to rotate during prepacking with 3x space occupied
	if optInnerRows*inputInnerCols*3 > params.LogSlots() {
		return optInnerRows
	} else {
		for optInnerRows*inputInnerCols*3 > params.LogSlots() {
			optInnerRows--
		}
		return optInnerRows
	}
}

//Compute tha average multiplication complexity of a series of splits
func GetAvgComplexity(splits []BlockSplits) float64 {
	complexity := 0
	n := splits[0].RowP
	for i := 0; i < len(splits)-1; i++ {
		m := splits[i].ColP
		l := splits[i+1].ColP
		complexity += n * m * l
	}
	return float64(complexity) / float64(len(splits)-1)
}

//Compute tha max multiplication complexity of a series of splits
func GetMaxComplexity(splits []BlockSplits) float64 {
	complexity := 0
	maxComplexity := complexity
	n := splits[0].RowP
	for i := 0; i < len(splits)-1; i++ {
		m := splits[i].ColP
		l := splits[i+1].ColP
		complexity = n * m * l
		if complexity > maxComplexity {
			maxComplexity = complexity
		}
	}
	return float64(maxComplexity)
}

// Finds all possible splits for the current model, given the number
// of input features (e.g 784 in case of MNIST) and the dimentions of all
// the weights of the model expressed as matrices.
// You can also provide the inputRows in case of testing where the number of rows is given, or set to -1 to let the splitter decide
// FindSplits will follow an heuristic approach, trying to find the best trade-off between throughput and complexity
func FindSplits(inputRows, inputFeatures int, weightRows, weightCols []int, params ckks.Parameters) [][]BlockSplits {
	var colPartitions []int
	var innerCols []int
	var batchSizes []int
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots())))

	sq := int(math.Ceil(math.Sqrt(float64(inputFeatures))))
	for d := 1; d <= sq; d++ {
		//look for all possible ways to divide the input features
		if inputFeatures%d == 0 {
			batch := int(math.Floor(slotsAvailable / (2 * float64(inputFeatures/d))))
			if inputRows != -1 {
				batch = utils.MinInt(batch, inputRows)
				for inputRows%batch != 0 {
					//resize to input rows divider
					batch--
				}
			}
			if batch*(inputFeatures/d)*2 <= int(slotsAvailable) {
				batchSizes = append(batchSizes, batch)
				colPartitions = append(colPartitions, d)
				innerCols = append(innerCols, inputFeatures/d)
			}
		}
	}

	//simulate to see if the splits work with the params

	var blockSplits [][]BlockSplits

	for i := range batchSizes {
		for filltresh := 0.9; filltresh >= 0.0; filltresh = filltresh - 0.1 {
			for _, strategyOnBatch := range []bool{true, false} {
				batch := batchSizes[i]
				colP := colPartitions[i]
				inCols := innerCols[i]
				currCols := inCols
				currColP := colP
				isValid := true

				blockSplit := make([]BlockSplits, len(weightRows)+1) //input + weights

				for w := range weightRows {
					inRowsW := weightRows[w] / currColP
					inColsW := plainUtils.Min(int(math.Floor(slotsAvailable/float64(inRowsW))), weightCols[w])
					for weightCols[w]%inColsW != 0 {
						inColsW--
					}
					for {
						//adjust weight inner col split or batch depending on strategy
						adjusted := true

						//check if weight submatrix can be stored and that the fillratio > threshold (also taking into account the max possible size of this weight)
						if inRowsW*inColsW > int(slotsAvailable) || GetFillRatio(inRowsW, inColsW, 1, float64(plainUtils.Min(int(slotsAvailable), weightRows[w]*weightCols[w]))) < filltresh {
							adjusted = false
						}

						//check if replication of input can be stored
						if GetReplicaFactor(inRowsW, inColsW)*batch*currCols > int(slotsAvailable) || GetFillRatio(batch, currCols, GetReplicaFactor(inRowsW, inColsW), slotsAvailable) < filltresh {
							adjusted = false
						}
						if !adjusted {
							if strategyOnBatch {
								//loop until divisor or lower than 1
								if inColsW <= 1 {
									isValid = false
									break
								} else {
									inColsW--
								}
								for weightCols[w]%inColsW != 0 && inColsW > 1 {
									inColsW--
								}
							} else {
								//decrease batch
								if batch <= 1 {
									isValid = false
									break
								} else if batch > 1 {
									batch--
								}
								if inputRows != -1 {
									for inputRows%batch != 0 && batch >= 1 {
										//resize to input rows divider
										batch--
									}
								}
							}
						} else {
							//is adjusted now
							break
						}
					}
					if !isValid {
						break
					}
					blockSplit[w+1] = BlockSplits{InnerRows: inRowsW, InnerCols: inColsW, RowP: currColP, ColP: weightCols[w] / inColsW}
					currColP = weightCols[w] / inColsW
					currCols = inColsW

					//repack
					if w < len(weightRows)-1 && currColP != 1 {
						nextInnerRowW := weightRows[w+1] / currColP
						nextInnerColW := plainUtils.Min(int(math.Floor(slotsAvailable/float64(nextInnerRowW))), weightCols[w+1])
						nextReplicaFactor := GetReplicaFactor(nextInnerRowW, nextInnerColW)

						//repack only if matrices are big 'enough'
						if float64(batch*currCols*nextReplicaFactor) > 0.5*slotsAvailable && float64(nextInnerRowW*nextInnerColW) > 0.5*slotsAvailable {
							//fmt.Println("Repacking...")
							possibleNewColP := make([]int, currCols)
							f := 0
							for div := 1; div <= currCols; div++ {
								if currCols%div == 0 {
									if currCols/div < currColP {
										possibleNewColP[f] = currCols / div
										f++
									}
								}
							}
							possibleNewColP = possibleNewColP[:f]

							for f := len(possibleNewColP) - 1; f >= 0; f-- {
								tmpColP := possibleNewColP[f]
								nextInnerRowW := weightRows[w+1] / tmpColP
								nextInnerColW := plainUtils.Min(int(math.Floor(slotsAvailable/float64(nextInnerRowW))), weightCols[w])
								nextReplicaFactor := GetReplicaFactor(nextInnerRowW, nextInnerColW)

								tmpCols := currCols * currColP / tmpColP
								if float64(batch*tmpCols*nextReplicaFactor) < slotsAvailable {
									//fmt.Println("Repacking done...")
									//fmt.Println("ColP was: ", currColP)
									currColP = tmpColP
									currCols = tmpCols
									//fmt.Println("ColP now: ", currColP)
									break
								}
							}
						}
					}
				}
				if isValid {
					//save this split
					var rowP int
					if inputRows == -1 {
						rowP = 1
					} else {
						rowP = inputRows / batch
					}
					blockSplit[0] = BlockSplits{InnerRows: batch, InnerCols: inCols, RowP: rowP, ColP: colP}
					blockSplits = append(blockSplits, blockSplit)
				}
			}
		}
	}
	if len(blockSplits) == 0 {
		return blockSplits
	}
	maxRatio := float64(blockSplits[0][0].InnerRows) / GetMaxComplexity(blockSplits[0])

	for i := 0; i < len(blockSplits); i++ {
		complexity := GetMaxComplexity(blockSplits[i])
		if float64(blockSplits[i][0].InnerRows)/complexity > maxRatio {
			maxRatio = float64(blockSplits[i][0].InnerRows) / complexity
		}
	}
	var filteredSplits [][]BlockSplits

	for i := 0; i < len(blockSplits); i++ {
		if float64(blockSplits[i][0].InnerRows)/GetMaxComplexity(blockSplits[i]) == maxRatio {
			filteredSplits = append(filteredSplits, blockSplits[i])
		}
	}
	if len(filteredSplits) > 1 {
		minColP := filteredSplits[0][0].ColP
		idxMinColP := 0
		for i := range filteredSplits {
			if filteredSplits[i][0].RowP < minColP {
				minColP = filteredSplits[i][0].ColP
				idxMinColP = i
			}
		}
		filteredSplits = [][]BlockSplits{filteredSplits[idxMinColP]}
	}
	return filteredSplits
}

//Exctract info useful for splitting input and weights, plus a hashcode for serialization
func ExctractInfo(splits []BlockSplits) (SplitsInfo, string) {
	info := SplitsInfo{}
	info.InputRows = splits[0].InnerRows
	info.InputCols = splits[0].InnerCols
	info.InputRowP, info.InputColP = splits[0].RowP, splits[0].ColP
	info.NumWeights = len(splits) - 1
	info.RowsOfWeights = make([]int, info.NumWeights)
	info.ColsOfWeights = make([]int, info.NumWeights)
	info.RowPOfWeights = make([]int, info.NumWeights)
	info.ColPOfWeights = make([]int, info.NumWeights)
	for i, split := range splits[1:] {
		info.RowsOfWeights[i] = split.InnerRows
		info.ColsOfWeights[i] = split.InnerCols
		info.RowPOfWeights[i] = split.RowP
		info.ColPOfWeights[i] = split.ColP
	}
	buf, _ := json.Marshal(info)
	hasher := sha1.New()
	hasher.Write(buf)
	code := base64.RawURLEncoding.EncodeToString(hasher.Sum(nil))
	return info, code
}

func PrintAllSplits(splits [][]BlockSplits) {
	for i := range splits {
		fmt.Printf("\nPossible split %d\n", i+1)
		PrintSetOfSplits(splits[i])
	}
}

func PrintSetOfSplits(setOfSplits []BlockSplits) {
	fmt.Println("-------------------------------------------------------------------------------------------")
	for j := range setOfSplits {
		var splittingWhat string
		if j == 0 {
			splittingWhat = "Input"
		} else {
			splittingWhat = fmt.Sprintf("Weight %d", j)
		}
		split := setOfSplits[j]
		fmt.Println("-------------------------------------------------------------")
		fmt.Println("Splits for ", splittingWhat)
		fmt.Printf("Inner Rows: %d Inner Cols: %d Row Partitions: %d Col Partitions: %d\n\n", split.InnerRows, split.InnerCols, split.RowP, split.ColP)
	}
	fmt.Println("-------------------------------------------------------------------------------------------")
}
