# forecast
В архиве несколько питоновских файлов.
1) data_preparation_1 - этот файл следует запускать для препроцессинга исходных данных. Там есть единственный параметр - глубина предыстории. Сейчас она составляет 24 часа. Дополнительно там применяется нормализация и дамми кодирование + генерируются бинарные ответы. После чего  это все сохраняется в новый файл input.csv. Также создается файл predict_features.csv для прогнозирования будущего ответа по самым последним данным.

2) build_model_deep_net_2 и build_model_LSTM_2 - это два скрипта для обучения глубокой нейронной сети и глубокой сети типа LSTM. Архитектуру и число итераций можите менять по Вашему усмотрению. Я сделал исходя из своих вычислительных возможностей.

3) predict_based_on_2_models_3 - это скрипт для прогнозирования будущего ответа по самым последним данным на основе 2 обученных сетей. Результат печатается в консоли.

Первая глубокая сеть (DNN) оказалась получе LSTM.


Для прогнозирования новых данных Вы должны сначала выполнить их препроцессинг скриптом из п.1, за тем запустить скрипт из пункта 3. Учтите, что формат должен быть аналогичным тому, что мы использовали сейчас: Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT
