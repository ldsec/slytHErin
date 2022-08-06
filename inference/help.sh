#!/bin/sh
echo "A new web page should open with the documentation. If not, check that you have godoc installed:"
echo "$ go get golang.org/x/tools/cmd/godoc"
python3 -m webbrowser -n http://localhost:6060/pkg/github.com/ldsec/dnn-inference/inference/ &
godoc
