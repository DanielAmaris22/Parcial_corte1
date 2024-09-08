import unittest
from ML_FLOW import ML_FLOW_PARCIAL

class prueba(unittest.TestCase):
    def test_model(self):
        modelo = ML_FLOW_PARCIAL(met=False, mod=1)
        salida = modelo.ML_FLOW()

        if salida["success"]==True:
            self.assertTrue(salida["success"],"Completed succesfully ...")
            self.assertGreaterEqual(salida["accuracy"],70,'Excellent performance')
            a = "Completed succesfully ..."
            b = salida["accuracy"]
            return {'Procces':a,'Accuracy':b}
        else:
            a = "Not completed succesfully ..."
            b = salida["message"]
            return {'Procces':a,',Message':b}

pb = prueba()
print(pb.test_model())

if __name__ == "_main_":
    unittest.main()