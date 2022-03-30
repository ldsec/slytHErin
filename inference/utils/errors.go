package utils

func ThrowErr(err error) {
	if err != nil {
		panic(err)
	}
}
