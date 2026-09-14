# !pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
ventilador = ctrl.Consequent(np.arange(0, 101, 1), 'ventilador')

temperatura['frio'] = fuzz.trimf(temperatura.universe, [0, 10, 25])
temperatura['calor'] = fuzz.trimf(temperatura.universe, [15, 30, 40])

ventilador['lento'] = fuzz.trimf(ventilador.universe, [0, 25, 60])
ventilador['rapido'] = fuzz.trimf(ventilador.universe, [40, 75, 100])

regla1 = ctrl.Rule(temperatura['frio'], ventilador['lento'])
regla2 = ctrl.Rule(temperatura['calor'], ventilador['rapido'])

sistema = ctrl.ControlSystem([regla1, regla2])

for t in [10, 15, 20, 25, 30, 35]:
    simulador = ctrl.ControlSystemSimulation(sistema)
    simulador.input['temperatura'] = t
    simulador.compute()
    print(f"Temperatura: {t} -> Ventilador: {simulador.output['ventilador']:.2f}")

temperatura.view()
ventilador.view()
