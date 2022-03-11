class Logger():
  def __init__(self, path="./logs", name="SimpleNet"):
    self.path = path
    self.name = name
    self.log = []
  
  def log_step(self, epoch, step, loss, accuracy):
    self.log.append(f"[!] Training Epoch {epoch+1}, step {step+1} ==> loss {loss}, accuracy {accuracy}")
  
  def log_batch(self, batch, loss, accuracy):
    self.log.append(f"[!] Test batch {batch+1} ==> loss {loss}, accuracy {accuracy}")
  
  def finalize(self, test_loss, test_accuracy):
    self.log.append("=================================")
    self.log.append(f"[+] Average test Loss ==> {test_loss:.4f}")
    self.log.append(f"[+] Test accuracy ==> {test_accuracy * 100:.2f}")
    with open(self.path+self.name+"_log.txt", "w+") as f:
      f.write("\n".join(self.log))
