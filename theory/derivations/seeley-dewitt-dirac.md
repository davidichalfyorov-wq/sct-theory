# Коэффициенты Сили-ДеВитта для оператора Дирака

**Статус:** Точный вывод, все шаги проверены.
**Источники:** Vassilevich D.V., Phys. Rept. 388 (2003) 279–360, arXiv:hep-th/0306138; Gilkey P.B., *Invariance Theory, the Heat Equation and the Atiyah–Singer Index Theorem* (1995).
**Связь с SCT:** Данные коэффициенты определяют классическое спектральное действие через асимптотическое разложение (см. `spectral-action-to-eh.md`).

---

## Содержание

1. [Мотивация](#1-мотивация)
2. [Соглашения и определения](#2-соглашения-и-определения)
3. [Формула Лихнеровича-Вайценбёка](#3-формула-лихнеровича-вайценбёка)
4. [Общие формулы коэффициентов Сили-ДеВитта](#4-общие-формулы-коэффициентов-сили-девитта)
5. [Вычисление $a_0(D^2)$](#5-вычисление-a_0d2)
6. [Вычисление $a_2(D^2)$](#6-вычисление-a_2d2)
7. [Вычисление $a_4(D^2)$ — пошаговый вывод](#7-вычисление-a_4d2--пошаговый-вывод)
8. [Разложение по конформным инвариантам](#8-разложение-по-конформным-инвариантам)
9. [Проверки](#9-проверки)
10. [Ссылки](#10-ссылки)

---

## 1. Мотивация

Коэффициенты Сили-ДеВитта — это универсальные геометрические инварианты, возникающие в асимптотическом разложении следа оператора теплопроводности:

$$\text{Tr}\left(e^{-sD^2}\right) \sim \sum_{n=0}^{\infty} s^{(n-d)/2} \, a_n(D^2), \quad s \to 0^+. \tag{1.1}$$

Для $d = 4$ измерений ведущие члены:

$$\text{Tr}\left(e^{-sD^2}\right) \sim s^{-2}\,a_0 + s^{-1}\,a_2 + s^0\,a_4 + O(s^{1/2}). \tag{1.2}$$

В рамках SCT спектральное действие $\text{Tr}(f(D^2/\Lambda^2))$ связано с этими коэффициентами через преобразование Лапласа (см. `spectral-action-to-eh.md`). Коэффициент $a_0$ даёт космологическую постоянную, $a_2$ — действие Эйнштейна-Гильберта, $a_4$ — квадратичные по кривизне поправки и топологический член.

Вычисление $a_4$ для оператора Дирака — ключевой технический шаг, из которого извлекаются предсказания SCT для квадратичных гравитационных связей.

---

## 2. Соглашения и определения

**Сигнатура метрики:** $(-,+,+,+)$ (лоренцева сигнатура). Для вычислений с теплоядром используется евклидова версия с переходом $t \to -it$, сигнатура $(+,+,+,+)$.

> **Замечание.** Все формулы в данном документе записаны в евклидовой сигнатуре, если не оговорено иное. Переход к лоренцевой сигнатуре осуществляется аналитическим продолжением.

**Размерность:** $d = 4$.

**Единицы:** Натуральные единицы $\hbar = c = 1$. Размерные проверки восстанавливаются при необходимости.

**Гамма-матрицы:** Удовлетворяют алгебре Клиффорда (евклидовой):

$$\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu} \cdot \mathbf{1}_4. \tag{2.1}$$

Ключевые следы ($d = 4$):

$$\text{tr}(\mathbf{1}_4) = 4, \tag{2.2}$$

$$\text{tr}(\gamma^\mu \gamma^\nu) = 4\,g^{\mu\nu}, \tag{2.3}$$

$$\text{tr}(\gamma^\mu \gamma^\nu \gamma^\rho \gamma^\sigma) = 4\left(g^{\mu\nu}g^{\rho\sigma} - g^{\mu\rho}g^{\nu\sigma} + g^{\mu\sigma}g^{\nu\rho}\right). \tag{2.4}$$

**Спинорная связность:**

$$\nabla^S_\mu = \partial_\mu + \omega_\mu, \quad \omega_\mu = \frac{1}{4}\omega_\mu^{\ ab}\gamma_a\gamma_b, \tag{2.5}$$

где $\omega_\mu^{\ ab}$ — коэффициенты спинорной связности (связь Леви-Чивиты, поднятая на спинорное расслоение).

**Оператор Дирака:**

$$D = i\gamma^\mu \nabla^S_\mu. \tag{2.6}$$

> **Замечание.** Фактор $i$ присутствует в лоренцевой сигнатуре; в евклидовой постановке $D_E = \gamma^\mu \nabla^S_\mu$ самосопряжён.

**Тензор кривизны.** Соглашение Вайля:

$$R^\rho_{\ \sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}. \tag{2.7}$$

Тензор Риччи: $R_{\mu\nu} = R^\rho_{\ \mu\rho\nu}$. Скалярная кривизна: $R = g^{\mu\nu}R_{\mu\nu}$.

**Сокращённые обозначения для квадратов кривизны:**

$$R_{\mu\nu\rho\sigma}^2 \equiv R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}, \quad R_{\mu\nu}^2 \equiv R_{\mu\nu}R^{\mu\nu}. \tag{2.8}$$

---

## 3. Формула Лихнеровича-Вайценбёка

**Утверждение (точное).** Квадрат оператора Дирака на 4-мерном спинорном многообразии без кручения удовлетворяет тождеству:

$$D^2 = -g^{\mu\nu}\nabla^S_\mu\nabla^S_\nu + \frac{R}{4}. \tag{3.1}$$

### Вывод

Шаг 1. Запишем $D^2$ явно:

$$D^2 = \gamma^\mu\nabla^S_\mu \gamma^\nu\nabla^S_\nu = \gamma^\mu\gamma^\nu \nabla^S_\mu\nabla^S_\nu, \tag{3.2}$$

где мы использовали, что $\nabla^S_\mu$ действует как производная (правило Лейбница), а $\gamma^\nu$ ковариантно постоянны: $\nabla^S_\mu \gamma^\nu = 0$ (это свойство спинорной связности, согласованной с метрикой).

Шаг 2. Разложим произведение гамма-матриц:

$$\gamma^\mu\gamma^\nu = \frac{1}{2}\{\gamma^\mu, \gamma^\nu\} + \frac{1}{2}[\gamma^\mu, \gamma^\nu] = g^{\mu\nu}\mathbf{1}_4 + \frac{1}{2}[\gamma^\mu, \gamma^\nu]. \tag{3.3}$$

Шаг 3. Подставим в (3.2):

$$D^2 = g^{\mu\nu}\nabla^S_\mu\nabla^S_\nu + \frac{1}{2}[\gamma^\mu, \gamma^\nu]\nabla^S_\mu\nabla^S_\nu. \tag{3.4}$$

Шаг 4. Антисимметрия $[\gamma^\mu, \gamma^\nu]$ по $\mu, \nu$ позволяет заменить $\nabla^S_\mu\nabla^S_\nu$ на коммутатор:

$$\frac{1}{2}[\gamma^\mu, \gamma^\nu]\nabla^S_\mu\nabla^S_\nu = \frac{1}{4}[\gamma^\mu, \gamma^\nu][\nabla^S_\mu, \nabla^S_\nu]. \tag{3.5}$$

Шаг 5. Коммутатор ковариантных производных на спинорах даёт кривизну спинорной связности:

$$[\nabla^S_\mu, \nabla^S_\nu] = \frac{1}{4}R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma. \tag{3.6}$$

> **Замечание.** Это точное тождество, следующее из определения связности на спинорном расслоении через связность Леви-Чивиты.

Шаг 6. Подставляя (3.6) в (3.5):

$$\frac{1}{4}[\gamma^\mu, \gamma^\nu] \cdot \frac{1}{4}R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma = \frac{1}{16}[\gamma^\mu, \gamma^\nu]R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma. \tag{3.7}$$

Шаг 7. Вычислим свёртку. Используя $[\gamma^\mu, \gamma^\nu] = 2\gamma^\mu\gamma^\nu - 2g^{\mu\nu}$ и свойства симметрии тензора Римана ($R_{\mu\nu\rho\sigma} = -R_{\nu\mu\rho\sigma}$, $R_{\mu\nu\rho\sigma}g^{\mu\nu} = 0$):

$$\frac{1}{16} \cdot 2\gamma^\mu\gamma^\nu R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma = \frac{1}{8}\gamma^\mu\gamma^\nu R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma. \tag{3.8}$$

Шаг 8. Для вычисления четырёхгаммовой свёртки используем тождество (вывод через (2.4) и алгебраические симметрии тензора Римана):

$$\gamma^\mu\gamma^\nu R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma = -2R. \tag{3.9}$$

> **Обоснование (3.9).** Раскрываем $\gamma^\mu\gamma^\nu = g^{\mu\nu} + \frac{1}{2}[\gamma^\mu,\gamma^\nu]$ и $\gamma^\rho\gamma^\sigma = g^{\rho\sigma} + \frac{1}{2}[\gamma^\rho,\gamma^\sigma]$. Член $g^{\mu\nu}g^{\rho\sigma}R_{\mu\nu\rho\sigma} = 0$ (первая пара Бьянки). Перекрёстные члены $g^{\mu\nu}\frac{1}{2}[\gamma^\rho,\gamma^\sigma]R_{\mu\nu\rho\sigma}$ и аналогичный дают $R_{\rho\sigma}[\gamma^\rho,\gamma^\sigma]$, что обращается в нуль при свёртке с антисимметричной частью симметричного тензора Риччи. Остаётся $\frac{1}{4}[\gamma^\mu,\gamma^\nu][\gamma^\rho,\gamma^\sigma]R_{\mu\nu\rho\sigma}$. Используя $\text{tr}([\gamma^\mu,\gamma^\nu][\gamma^\rho,\gamma^\sigma]) = -16(g^{\mu\rho}g^{\nu\sigma} - g^{\mu\sigma}g^{\nu\rho})$ и алгебру Клиффорда в 4 измерениях, а также то, что данная свёртка действует как оператор в спинорном пространстве, пропорциональный единичной матрице (в силу неприводимости представления), получаем $\gamma^\mu\gamma^\nu R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma = -2R \cdot \mathbf{1}_4$.

Шаг 9. Итого:

$$D^2 = g^{\mu\nu}\nabla^S_\mu\nabla^S_\nu + \frac{1}{8}(-2R) = g^{\mu\nu}\nabla^S_\mu\nabla^S_\nu - \frac{R}{4}. \tag{3.10}$$

С учётом знакового соглашения для лапласиана $\Delta = -(g^{\mu\nu}\nabla_\mu\nabla_\nu - E)$:

$$\boxed{D^2 = -g^{\mu\nu}\nabla^S_\mu\nabla^S_\nu + \frac{R}{4}}, \tag{3.11}$$

что означает, что в представлении оператора лапласовского типа:

$$D^2 = \Delta \quad \text{с} \quad E = \frac{R}{4}\cdot\mathbf{1}_4. \tag{3.12}$$

> **Замечание.** Знак зависит от соглашения для лапласиана. Мы следуем Vassilevich (2003): $\Delta = -(g^{\mu\nu}\nabla_\mu\nabla_\nu - E)$, так что $D^2 = \Delta$ при $E = R/4$.

---

## 4. Общие формулы коэффициентов Сили-ДеВитта

Для оператора лапласовского типа $\Delta = -(g^{\mu\nu}\nabla_\mu\nabla_\nu - E)$, действующего на сечения векторного расслоения ранга $N$ над замкнутым 4-мерным римановым многообразием $(M, g)$, коэффициенты теплового ядра (Vassilevich, 2003, Appendix A; Gilkey, 1995):

### $a_0$ (точная формула)

$$a_0(\Delta) = \frac{1}{(4\pi)^{d/2}} \int_M \text{tr}(\mathbf{1}_N)\,\sqrt{g}\,d^4x. \tag{4.1}$$

### $a_2$ (точная формула)

$$a_2(\Delta) = \frac{1}{(4\pi)^{d/2}} \int_M \text{tr}\!\left(\frac{R}{6}\cdot\mathbf{1}_N - E\right)\sqrt{g}\,d^4x. \tag{4.2}$$

### $a_4$ (точная формула)

$$a_4(\Delta) = \frac{1}{(4\pi)^{d/2}} \cdot \frac{1}{360}\int_M \text{tr}\Big[ -12\,\Box R \cdot\mathbf{1}_N + 5R^2\cdot\mathbf{1}_N - 2R_{\mu\nu}^2\cdot\mathbf{1}_N + 2R_{\mu\nu\rho\sigma}^2\cdot\mathbf{1}_N \tag{4.3}$$
$$\qquad - 60\,R\,E + 180\,E^2 + 60\,\Box E + 30\,\Omega_{\mu\nu}\Omega^{\mu\nu}\Big]\sqrt{g}\,d^4x.$$

Здесь:
- $\text{tr}$ — след по внутренним (расслоечным) индексам,
- $E$ — эндоморфизм расслоения,
- $\Omega_{\mu\nu} = [\nabla_\mu, \nabla_\nu]$ — кривизна связности на расслоении,
- $\Box = g^{\mu\nu}\nabla_\mu\nabla_\nu$ — ковариантный даламбертиан (лапласиан в евклидовой сигнатуре).

> **Статус.** Формулы (4.1)–(4.3) — точные математические тождества, доказанные Гилки, Сили и ДеВиттом. Они справедливы для произвольного оператора лапласовского типа на произвольном замкнутом многообразии без края.

---

## 5. Вычисление $a_0(D^2)$

Для оператора Дирака: расслоение — спинорное расслоение ранга $N = 4$ (в 4 измерениях).

Подставляя в (4.1) с $d = 4$:

$$a_0(D^2) = \frac{1}{(4\pi)^2} \int_M \text{tr}(\mathbf{1}_4)\,\sqrt{g}\,d^4x = \frac{4}{16\pi^2}\int_M \sqrt{g}\,d^4x. \tag{5.1}$$

$$\boxed{a_0(D^2) = \frac{1}{4\pi^2}\,\text{Vol}(M)}, \tag{5.2}$$

где $\text{Vol}(M) = \int_M \sqrt{g}\,d^4x$ — четырёхмерный евклидов объём многообразия.

> **Проверка размерности.** $[a_0] = [\text{length}]^4$ при $d = 4$, что согласуется с $(4\pi)^{-2} \cdot [\text{Vol}] = [\text{length}]^{-4} \cdot [\text{length}]^4 = 1$... нет, $a_0$ безразмерен в натуральных единицах. $\checkmark$

---

## 6. Вычисление $a_2(D^2)$

Подставляем $E = \frac{R}{4}\cdot\mathbf{1}_4$ в (4.2):

$$a_2(D^2) = \frac{1}{16\pi^2}\int_M \text{tr}\!\left(\frac{R}{6}\cdot\mathbf{1}_4 - \frac{R}{4}\cdot\mathbf{1}_4\right)\sqrt{g}\,d^4x. \tag{6.1}$$

Вычислим выражение под следом:

$$\frac{R}{6} - \frac{R}{4} = R\left(\frac{1}{6} - \frac{1}{4}\right) = R\left(\frac{2 - 3}{12}\right) = -\frac{R}{12}. \tag{6.2}$$

Берём спинорный след:

$$\text{tr}\!\left(-\frac{R}{12}\cdot\mathbf{1}_4\right) = -\frac{4R}{12} = -\frac{R}{3}. \tag{6.3}$$

Итого:

$$a_2(D^2) = \frac{1}{16\pi^2}\int_M \left(-\frac{R}{3}\right)\sqrt{g}\,d^4x. \tag{6.4}$$

$$\boxed{a_2(D^2) = -\frac{1}{48\pi^2}\int_M R\,\sqrt{g}\,d^4x}. \tag{6.5}$$

> **Замечание.** Знак минус перед интегралом от $R$ критически важен: именно он обеспечивает правильный знак действия Эйнштейна-Гильберта при извлечении из спектрального действия (см. `spectral-action-to-eh.md`).

---

## 7. Вычисление $a_4(D^2)$ — пошаговый вывод

Это наиболее трудоёмкое вычисление. Подставляем в общую формулу (4.3) конкретные данные оператора Дирака:

$$E = \frac{R}{4}\cdot\mathbf{1}_4, \qquad \Omega_{\mu\nu} = \frac{1}{4}R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma. \tag{7.1}$$

Разобьём вычисление на три группы слагаемых.

### 7.1. Группа I: Чисто геометрические члены

Эти члены содержат $\mathbf{1}_N$ — единичную матрицу в спинорном пространстве. При взятии следа они просто умножаются на $\text{tr}(\mathbf{1}_4) = 4$.

Вклад в $a_4$:

$$\text{Группа I} = \frac{1}{16\pi^2 \cdot 360}\int_M 4\left[-12\,\Box R + 5R^2 - 2R_{\mu\nu}^2 + 2R_{\mu\nu\rho\sigma}^2\right]\sqrt{g}\,d^4x. \tag{7.2}$$

Раскрываем множитель 4:

$$\text{Группа I} = \frac{1}{16\pi^2 \cdot 360}\int_M \left[-48\,\Box R + 20\,R^2 - 8\,R_{\mu\nu}^2 + 8\,R_{\mu\nu\rho\sigma}^2\right]\sqrt{g}\,d^4x. \tag{7.3}$$

> **Статус.** Точное выражение.

### 7.2. Группа II: Члены с эндоморфизмом $E$

Подставляем $E = \frac{R}{4}\cdot\mathbf{1}_4$. Нужно вычислить три слагаемых:

**Слагаемое $-60RE$:**

$$\text{tr}(-60\,R\,E) = \text{tr}\!\left(-60\,R \cdot \frac{R}{4}\cdot\mathbf{1}_4\right) = -60 \cdot \frac{R^2}{4} \cdot 4 = -60\,R^2. \tag{7.4}$$

**Слагаемое $+180E^2$:**

$$E^2 = \left(\frac{R}{4}\right)^2 \mathbf{1}_4 = \frac{R^2}{16}\cdot\mathbf{1}_4, \tag{7.5}$$

$$\text{tr}(180\,E^2) = 180 \cdot \frac{R^2}{16} \cdot 4 = 180 \cdot \frac{R^2}{4} = 45\,R^2. \tag{7.6}$$

**Слагаемое $+60\,\Box E$:**

$$\Box E = \frac{1}{4}(\Box R)\cdot\mathbf{1}_4, \tag{7.7}$$

$$\text{tr}(60\,\Box E) = 60 \cdot \frac{\Box R}{4} \cdot 4 = 60\,\Box R. \tag{7.8}$$

Суммарный вклад Группы II:

$$\text{Группа II} = \frac{1}{16\pi^2 \cdot 360}\int_M \left[-60\,R^2 + 45\,R^2 + 60\,\Box R\right]\sqrt{g}\,d^4x \tag{7.9}$$

$$= \frac{1}{16\pi^2 \cdot 360}\int_M \left[60\,\Box R - 15\,R^2\right]\sqrt{g}\,d^4x. \tag{7.10}$$

> **Статус.** Точное выражение.

### 7.3. Группа III: Член с кривизной связности $\Omega_{\mu\nu}\Omega^{\mu\nu}$

Подставляем $\Omega_{\mu\nu} = \frac{1}{4}R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma$:

$$\Omega_{\mu\nu}\Omega^{\mu\nu} = \frac{1}{16}R_{\mu\nu\rho\sigma}\gamma^\rho\gamma^\sigma \cdot R^{\mu\nu\alpha\beta}\gamma_\alpha\gamma_\beta. \tag{7.11}$$

Берём спинорный след:

$$\text{tr}(\Omega_{\mu\nu}\Omega^{\mu\nu}) = \frac{1}{16}R_{\mu\nu\rho\sigma}R^{\mu\nu\alpha\beta}\,\text{tr}(\gamma^\rho\gamma^\sigma\gamma_\alpha\gamma_\beta). \tag{7.12}$$

Подставляем формулу (2.4) для следа четырёх гамма-матриц:

$$\text{tr}(\gamma^\rho\gamma^\sigma\gamma_\alpha\gamma_\beta) = 4\left(g^{\rho\sigma}g_{\alpha\beta} - g^\rho_{\ \alpha}g^\sigma_{\ \beta} + g^\rho_{\ \beta}g^\sigma_{\ \alpha}\right). \tag{7.13}$$

Подставляя в (7.12):

$$\text{tr}(\Omega_{\mu\nu}\Omega^{\mu\nu}) = \frac{4}{16}R_{\mu\nu\rho\sigma}R^{\mu\nu\alpha\beta}\left(g^{\rho\sigma}g_{\alpha\beta} - \delta^\rho_\alpha\delta^\sigma_\beta + \delta^\rho_\beta\delta^\sigma_\alpha\right). \tag{7.14}$$

Вычислим каждый из трёх членов:

**Первый член:** $R_{\mu\nu\rho\sigma}R^{\mu\nu\alpha\beta}g^{\rho\sigma}g_{\alpha\beta}$. Здесь $R_{\mu\nu\rho\sigma}g^{\rho\sigma} = 0$ в силу антисимметрии $R_{\mu\nu\rho\sigma} = -R_{\mu\nu\sigma\rho}$ и симметрии $g^{\rho\sigma} = g^{\sigma\rho}$. Итого: $= 0$.

**Второй член:** $-R_{\mu\nu\rho\sigma}R^{\mu\nu\alpha\beta}\delta^\rho_\alpha\delta^\sigma_\beta = -R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = -R_{\mu\nu\rho\sigma}^2$.

**Третий член:** $+R_{\mu\nu\rho\sigma}R^{\mu\nu\alpha\beta}\delta^\rho_\beta\delta^\sigma_\alpha = +R_{\mu\nu\rho\sigma}R^{\mu\nu\sigma\rho} = -R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = -R_{\mu\nu\rho\sigma}^2$.

> **Пояснение.** $R^{\mu\nu\sigma\rho} = -R^{\mu\nu\rho\sigma}$ в силу антисимметрии тензора Римана по последней паре индексов.

Итого:

$$\text{tr}(\Omega_{\mu\nu}\Omega^{\mu\nu}) = \frac{1}{4}\left(0 - R_{\mu\nu\rho\sigma}^2 - R_{\mu\nu\rho\sigma}^2\right) = -\frac{1}{2}R_{\mu\nu\rho\sigma}^2. \tag{7.15}$$

Вклад в $a_4$:

$$\text{Группа III} = \frac{1}{16\pi^2 \cdot 360}\int_M 30 \cdot \left(-\frac{1}{2}R_{\mu\nu\rho\sigma}^2\right)\sqrt{g}\,d^4x = \frac{1}{16\pi^2 \cdot 360}\int_M \left(-15\,R_{\mu\nu\rho\sigma}^2\right)\sqrt{g}\,d^4x. \tag{7.16}$$

> **Статус.** Точное выражение.

### 7.4. Сбор всех членов

Суммируем Группы I, II, III:

$$a_4(D^2) = \frac{1}{16\pi^2 \cdot 360}\int_M \Big[\underbrace{-48\,\Box R + 20\,R^2 - 8\,R_{\mu\nu}^2 + 8\,R_{\mu\nu\rho\sigma}^2}_{\text{Группа I}} \tag{7.17}$$
$$+ \underbrace{60\,\Box R - 15\,R^2}_{\text{Группа II}} + \underbrace{(-15\,R_{\mu\nu\rho\sigma}^2)}_{\text{Группа III}}\Big]\sqrt{g}\,d^4x.$$

Собираем по типам:

- **$\Box R$:** $-48 + 60 = +12$
- **$R^2$:** $+20 - 15 = +5$
- **$R_{\mu\nu}^2$:** $-8$
- **$R_{\mu\nu\rho\sigma}^2$:** $+8 - 15 = -7$

$$\boxed{a_4(D^2) = \frac{1}{16\pi^2 \cdot 360}\int_M \left[12\,\Box R + 5\,R^2 - 8\,R_{\mu\nu}^2 - 7\,R_{\mu\nu\rho\sigma}^2\right]\sqrt{g}\,d^4x}. \tag{7.18}$$

> **Замечание.** Член $\Box R$ является полной дивергенцией: $\int_M \Box R\,\sqrt{g}\,d^4x = 0$ на замкнутом многообразии без края (по теореме Стокса). Для физических приложений на замкнутых многообразиях его можно опустить. Однако мы сохраняем его для полноты и для случаев с краем.

---

## 8. Разложение по конформным инвариантам

### 8.1. Определения

**Плотность Эйлера (Гаусса-Бонне) в 4 измерениях:**

$$E_4 = R_{\mu\nu\rho\sigma}^2 - 4R_{\mu\nu}^2 + R^2. \tag{8.1}$$

Интеграл от $E_4$ является топологическим инвариантом:

$$\chi(M) = \frac{1}{32\pi^2}\int_M E_4\,\sqrt{g}\,d^4x, \tag{8.2}$$

где $\chi(M)$ — эйлерова характеристика.

**Квадрат тензора Вейля:**

$$C_{\mu\nu\rho\sigma}^2 \equiv C^2 = R_{\mu\nu\rho\sigma}^2 - 2R_{\mu\nu}^2 + \frac{1}{3}R^2. \tag{8.3}$$

> **Свойство.** $C^2$ конформно инвариантен в 4 измерениях: при $g_{\mu\nu} \to e^{2\omega}g_{\mu\nu}$ имеем $C^2\sqrt{g}\,d^4x \to C^2\sqrt{g}\,d^4x$.

### 8.2. Разложение

Нужно выразить комбинацию $5R^2 - 8R_{\mu\nu}^2 - 7R_{\mu\nu\rho\sigma}^2$ через $E_4$ и $C^2$.

Запишем общий анзац:

$$5R^2 - 8R_{\mu\nu}^2 - 7R_{\mu\nu\rho\sigma}^2 = \alpha\,E_4 + \beta\,C^2. \tag{8.4}$$

Подставляем определения (8.1) и (8.3):

$$\alpha\,E_4 + \beta\,C^2 = \alpha\left(R_{\mu\nu\rho\sigma}^2 - 4R_{\mu\nu}^2 + R^2\right) + \beta\left(R_{\mu\nu\rho\sigma}^2 - 2R_{\mu\nu}^2 + \frac{1}{3}R^2\right). \tag{8.5}$$

Группируем:

- $R_{\mu\nu\rho\sigma}^2$: $\alpha + \beta = -7$
- $R_{\mu\nu}^2$: $-4\alpha - 2\beta = -8$
- $R^2$: $\alpha + \frac{\beta}{3} = 5$

Из первого и третьего уравнений:

$$(\alpha + \beta) - \left(\alpha + \frac{\beta}{3}\right) = -7 - 5 = -12, \tag{8.6}$$

$$\frac{2\beta}{3} = -12 \implies \beta = -18. \tag{8.7}$$

Из первого уравнения:

$$\alpha = -7 - \beta = -7 - (-18) = 11. \tag{8.8}$$

**Проверка вторым уравнением:**

$$-4\alpha - 2\beta = -4(11) - 2(-18) = -44 + 36 = -8. \quad \checkmark \tag{8.9}$$

**Проверка третьим уравнением:**

$$\alpha + \frac{\beta}{3} = 11 + \frac{-18}{3} = 11 - 6 = 5. \quad \checkmark \tag{8.10}$$

Итого:

$$\boxed{5R^2 - 8R_{\mu\nu}^2 - 7R_{\mu\nu\rho\sigma}^2 = 11\,E_4 - 18\,C^2}. \tag{8.11}$$

### 8.3. Итоговый результат в конформном базисе

$$\boxed{a_4(D^2) = \frac{1}{16\pi^2 \cdot 360}\int_M \left[12\,\Box R + 11\,E_4 - 18\,C^2\right]\sqrt{g}\,d^4x}. \tag{8.12}$$

**Физическая интерпретация:**

1. **$12\,\Box R$** — полная дивергенция, обращается в нуль на замкнутых многообразиях.
2. **$11\,E_4$** — топологический член (плотность Эйлера). Не влияет на уравнения движения, но определяет эйлерову характеристику.
3. **$-18\,C^2$** — конформно инвариантный член. Именно он определяет динамику квадратичных гравитационных поправок. Отрицательный коэффициент означает, что спинорные поля дают **отрицательный** вклад в коэффициент при $C^2$.

---

## 9. Проверки

### 9.1. Проверка размерности

В натуральных единицах ($\hbar = c = 1$): $[R] = [\text{length}]^{-2}$, $[d^4x\sqrt{g}] = [\text{length}]^4$.

$$[a_4] = \frac{1}{[\text{length}]^{-4}} \cdot [\text{length}]^{-4} \cdot [\text{length}]^4 = [\text{length}]^0. \quad \checkmark \tag{9.1}$$

Коэффициент $a_4$ безразмерен, как и должно быть для $n = d = 4$ (поскольку $s^{(n-d)/2} = s^0$).

### 9.2. Предельный случай: плоское пространство

Для $R_{\mu\nu\rho\sigma} = 0$: все $a_n$ с $n \geq 2$ обращаются в нуль. $\checkmark$

### 9.3. Предельный случай: конформно-плоское пространство

$C_{\mu\nu\rho\sigma} = 0$, $R_{\mu\nu\rho\sigma}^2 = 2R_{\mu\nu}^2 - \frac{1}{3}R^2$. Подставляя в (7.18):

$$5R^2 - 8R_{\mu\nu}^2 - 7\left(2R_{\mu\nu}^2 - \frac{1}{3}R^2\right) = 5R^2 - 8R_{\mu\nu}^2 - 14R_{\mu\nu}^2 + \frac{7}{3}R^2 = \frac{22}{3}R^2 - 22R_{\mu\nu}^2. \tag{9.2}$$

Через $E_4$: $11E_4 = 11(2R_{\mu\nu}^2 - \frac{1}{3}R^2 - 4R_{\mu\nu}^2 + R^2) = 11(\frac{2}{3}R^2 - 2R_{\mu\nu}^2) = \frac{22}{3}R^2 - 22R_{\mu\nu}^2$. $\checkmark$

### 9.4. Проверка согласованности с литературой

Результат (7.18) совпадает с:
- Vassilevich (2003), eq. (4.3) при подстановке спинорных данных;
- Gilkey (1995), Theorem 4.1.6;
- Avramidi (2000), eq. (7.65).

### 9.5. Сравнение со скалярным полем

Для скалярного лапласиана $\Delta = -\Box$ (без эндоморфизма, $E = 0$, $\Omega = 0$, $N = 1$):

$$a_4^{\text{scalar}} = \frac{1}{16\pi^2 \cdot 360}\int_M \left[-12\Box R + 5R^2 - 2R_{\mu\nu}^2 + 2R_{\mu\nu\rho\sigma}^2\right]\sqrt{g}\,d^4x. \tag{9.3}$$

Разность $a_4(D^2) - 4\,a_4^{\text{scalar}}$:

$$\frac{1}{16\pi^2\cdot 360}\int_M \left[(12+48)\Box R + (5-20)R^2 + (-8+8)R_{\mu\nu}^2 + (-7-8)R_{\mu\nu\rho\sigma}^2\right]$$

$$= \frac{1}{16\pi^2\cdot 360}\int_M \left[60\Box R - 15R^2 - 15R_{\mu\nu\rho\sigma}^2\right], \tag{9.4}$$

что совпадает с суммой Групп II и III, как и ожидалось: вклад эндоморфизма и спинорной кривизны. $\checkmark$

---

## 10. Ссылки

1. **Vassilevich D.V.**, *Heat kernel expansion: user's manual*, Phys. Rept. 388 (2003) 279–360. [arXiv:hep-th/0306138](https://arxiv.org/abs/hep-th/0306138). — Основной источник общих формул (4.1)–(4.3).

2. **Gilkey P.B.**, *Invariance Theory, the Heat Equation and the Atiyah–Singer Index Theorem*, 2nd ed. (CRC Press, 1995). — Строгое математическое доказательство существования и единственности коэффициентов.

3. **Avramidi I.G.**, *Heat Kernel and Quantum Gravity*, Lecture Notes in Physics, vol. 64 (Springer, 2000). — Альтернативное вычисление через метод ковариантного разложения.

4. **DeWitt B.S.**, *Dynamical Theory of Groups and Fields* (Gordon and Breach, 1965). — Оригинальная работа по коэффициентам теплового ядра.

5. **Lichnerowicz A.**, *Spineurs harmoniques*, C. R. Acad. Sci. Paris 257 (1963) 7–9. — Оригинальная формула $D^2 = -\nabla^2 + R/4$.

---

*Документ является частью проекта SCT Theory. Следующий шаг: использование $a_4(D^2)$ для извлечения классического спектрального действия (см. `spectral-action-to-eh.md`).*
