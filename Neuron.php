<?php

/*
    Easy nEural Library
    Copyright © 2018  Georgiy Aleksandrov (GeorgNation)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    -----------------------

    Contacts:

    Email: <gnation784@gmail.com>
    	   <mountain_blur228@bk.ru>
    VK:    vk.com/hacktriemackteam228
*/

define ('SEL_FUNCTION_LINEAR', 1); // Simple text
define ('SEL_FUNCTION_HYPERBOLIC_TANH', 2); // Гиперболический тангенс
define ('SEL_FUNCTION_ARCTG', 4); // Нахрен формулы! Арктангенс решит все проблемы! (нет)
define ('SEL_FUNCTION_SIGMOID', 8); // Надоевшая всем сигмоида но рекомендуемая
define ('SEL_FUNCTION_SQRT_SIGMOID', 16); // Теперь ее превратили в квадрат
define ('SEL_FUNCTION_LOGISTIC', 32); // Какая-то логистическая сигмоидальная функция. А на нее есть вообще применение?

class NetworkPCT_Simplified
{

	public $stack;

	public function newNetwork ($stacks, $inputs) // Создаем новую нейронную сеть
	{
		for ($i = 0; $i < $stacks; ++$i) // Создаем много стеков с нейронами.
		{
			$this->stack[] = new NetworkPCT_Simplified_Stack; // Создаем класс со стеком нейронов.
			for ($j = 0; $j < $inputs; ++$j) // Генерируем входы для стека
			{ 
				$this->stack[$i]->inputs[$j] = 0;
				$this->stack[$i]->weight[$j] = 2 * random_float (1, 3) - 1; // Так-как использования одного веса критично для нейросети, мы сгенерируем новое значение.
			}
			$this->stack[$i]->activF = SEL_FUNCTION_SIGMOID;
		}
	}

	public function activateFunction ($const)
	{
		if (defined ($const))
			throw new Exception ('Activate Function isn\'t defined. Активационная функция не объявлена.'); // Использование активационных функций, чьи не были инициализированы в скрипте, могут сделать нейросеть неработоспособной.

		if (!$this->stack)
			throw new Exception ('The function couldn\'t execute, because the neural network is not initialized. Функция не может быть выполнена, поскольку нейронная сеть не инициализирована.'); // Избегаем ошибки.

		for ($j = 0; $j < sizeof ($this->stack); ++$j)
		{ 
			$this->stack[$j]->activF = $const;
		}
	}

	public function train ($stackId, $input, $output, $ages = 10000)
	{

		$i = -1; // так надо
		$size_network = sizeof ($this->stack[$stackId]->inputs); // Размер входов в стеке
		$weights = $this->stack[$stackId]->weight; // Веса нейронов

		if (sizeof ($input) == sizeof ($this->stack[$stackId]->weight)) // Заверяемся, что размеры этих массивов сходятся
		{
			// Начинаем обучение.
			for ($age = 0; $age < $ages; ++$age) // В цикле проводим обучение
			{

				foreach ($input as $neo)
				{
					++$i; // Так надо

					$retrainedWeights = null; // Не хотим 2160 весов, обнуляем веса

					$train_input = $neo; // Храним вход
					$train_output = $this->smartActivation ($stackId, floatval ("{$train_input}.{$weights[$i]}")); // Вызываем активационную функцию в зависимости от выбора

					$adjust = $this->getAdjust ($output[$i], $train_output); // Находим ошибку сети

					foreach ($this->stack[$stackId]->weight as $w) // Проходим по весам сети
					{
						$retrainedWeights[] = $w + $adjust; // Корректируем вес
					}

					$this->stack[$stackId]->weight += $retrainedWeights; // Теперь складываем веса
				}

				$i = -1;
			}
		}
	}

	public function solve ($sid, $input) // Заставляем стек шевелить мозгами
	{
		$weights = $this->stack[$sid]->weight; // Переносим весы со стека в удобное место.
		$log = array ();                                                           // Логи
		if (sizeof ($input) == sizeof ($weights))                  // Никакого дисбаланса.
		{
			foreach ($weights as $weight) // Теперь проходимся по этим весам.....
			{
				
				$log[] = $this->smartActivation ($sid, floatval ("{$input[$i]}.{$weight}"));
				//$log[] = floatval ("{$input[$i]}.{$weight}");
			}
		}

		return $log;
	}

	protected function getAdjust ($actual, $excepted) // Получаем дополнение к весу
	{
		$error    = $excepted - $actual;   // Находим ошибку (разницу результатов сети)
		$adjust_1 = $excepted;
		$adjust_2 = $error * ($excepted * (1 - $actual));   // Вычисляем формулу.......
		$adjust   = floatval ("{$adjust_1}.{$adjust_2}"); // Делаем это число плавающим

		return $adjust; // Возвращаем ошибку сети
	}


	protected function smartActivation ($sid, $n)
	{
		$activation = $this->stack[$sid]->activF;
		switch ($activation)
		{
			case 1:
				$result = linear ($n);
			break;

			case 2:
				$result = hyperbolic ($n);
			break;
				
			case 4:
				$result = arctang ($n);
			break;
				
			case 8:
				$result = sigmoid ($n);
			break;
				
			case 16:
				$result = sqrt_sigmoid ($n);
			break;
				
			case 32:
				$result = logistic ($n);
			break;	
		}

		return $result;
	}

}

class NetworkPCT_Simplified_Stack
{
	public $activF;
	public $inputs;
	public $weight;
}

function random_float ($min, $max) // Генератор чисел с плавающей точкой
{
    return mt_rand ($min, $max - 1) + (mt_rand (0, PHP_INT_MAX - 1) / PHP_INT_MAX );
}

function linear ($n, $limit = 0.5) // Линейная функция (Жесткая пороговая функция)
{
	return ($n > $limit) ? 1 : 0;
}

function hyperbolic ($n, $e = 1) // Гиперболический тангенс (автор этой либы хер знает что это такое)
{
	return tanh ($n / $e);
}

function arctang ($n) // Арктангенс
{
	return atan ($n);
}

function sigmoid ($n) // Сигмоида
{
	return 1 / (1 + exp ($n * -1));
}

function sqrt_sigmoid ($n) // Квадратная сигмоида
{
	return 1 / sqrt (1 + exp ($n * -1));
}

function logistic ($n) // Логистическая функция
{
	return pow ((1 + exp (-$n)), -1);
}
