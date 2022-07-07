# kmooc 언어학 요약

- 언어학 기본 지식 : 학문 세부분야
    - 소리(음성,음운) → 형태(단어) → 구조(문장) → 의미
    - 음성 : 화자(조음음성학) → 공기 진동(음향음성학) → 청자 → 음성으로 파악(청취음성학)
    - 음성(실제 소리) ↔ 음소(소리에 대한 모델 ex. 한글 자모)
    - 말 뜻을 알아야 운율 정보를 반영하여 고품질 음성합성이 가능하여, 음성 뿐 아니라 형태 → 구조 → 의미 부분도 모델링이 필요함.
    - ex ) 음성대화
        - 사용자 음성 → 받아쓰기(음성인식) → 뜻 이해(음성언어이해) → Output(대화처리)
    
    <aside>
    💡 언어는 소리 → 형태 → 구조 → 의미의 계층 구조로 이루어져 있으며,
    단순 소리-글 간 변환은 음성만 다뤄도 충분하지만,
    어조, 억양, (특히 가창 합성의 경우)감정 등을 반영한 고품질의 음성 합성을 위해서는 
    형태 → 구조 → 의미를 다루는 모델 또한 필요함.
    
    </aside>
    

- 음성 신호 처리 기초 ( Wav → Spectrogram )
    - 신호처리란? + DFT
        
        <aside>
        💡 **요약**
        //요약하기
        
        </aside>
        
        1. 신호처리는 다음과 같은 두 부분으로 이루어 짐
            - **A/D Conversion** : 아날로그 → (Sampling, Quantization) →  디지털신호
                - **Sampling** : 시간(x축)이 Discrete해야 함 → $nT_s$
                - **Quantization** : Amplitude(진폭, 소리크기, y축)도 Discrete 해야 함 → $2^n$ 개 값 bin
            - 디지털신호 → ( 여러 좋은 feature 추출 ) → 의미 있는 데이터
            
        2. Linear Time-Invariant System
            - x(t) → [System T] → y(t), 즉  $T(x(t)) = y(t)$ 라 할 때, 이 system T를 **Linear**하다고 가정
                - T(ax+by) = aT(x) + bT(y)
            - **Time- Invariant**
                - $y(n-n_0) = T\left\{x(n-n_0) \right\}$
                - 시간에 영향(?)을 받지 않음
                - 미래 시간에 결과를 사용할 때, 단순 shift만 됐다고 볼 수 있어 시간에 따른 변화 고려 안 해도 됨.
            - $\delta$를 도입하여, $y(n) = x(n)*h(n)$ 얻음 → *는 Convolution Sum
            - 위의 식을 time system이 아닌 주파수 영역의 system에서 표현하면
                
                $$
                Y(k) = X(k)H(k)
                $$
                
                을 얻을 수 있다. ( Frequency domain에서는 합성곱이 곱으로 표현됨 )
                
        3. **Frequency** domain signal
            - 신호의 종류
                - time domain
                - frequency domain
            - time domain → frequency domain
                - fourier transform
                - 여러 주파수들로 신호를 분해(근사)
            
            | signal | Periodic | Aperiodic |
            | --- | --- | --- |
            | Continuous time | Fourier Series | Fourier Transform |
            | Discrete time | Descrete Fourier Series | DTFT, DFT |
            - DFT : 시간 및 주파수도 discrete ← 실제로 이걸 씀.
            
            $$
            f(t) = \Sigma_{i=0}^{\infty}a_ig_i(t)
            $$
            
            - $a_i$(eigenvalue) : 주파수 $g_i$(eigenfunction)에 해당하는 값
            - Frequency Domain의
            
            $$
            Y_k = X_kH_k
            $$
            
        
        의 관계에서, H는 필터 역할을 하게 된다.
        
        - ex) $H$가 특정 주파수 아래에서는 1, 그 위에는 0 값을 가지는 함수이면, 이는 특정 주파수 위를 날려버리는 필터로 사용될 수 있음
        
    - Nyquist Sampling
        
        ## Nyquist Sampling
        
        <aside>
        💡 **요약**
        
        </aside>
        
        - 연속 신호를 이산 신호로 샘플링을 할 때, 당연히 샘플링을 하는 주기가 길면(즉 속도가 느리면) 그 만큼 소실되는 정보가 많다.( 짧으면 비교적 많은 정보를 담을 수 있음 )
        - 어느 정도 빠르기 이상으로 샘플링하면, 소실된 값을 복원 가능하다
        
        <aside>
        💡 Real time domain signal에서 얻은 삼각파의 frequency domain signal의 주파수를
        $\omega b$라 하고, sampling frequency를 $\omega s$라 하면, $\omega s > 2\omega b$ 이면 복원 가능
        
        </aside>
        
        - by low pass filter
        - (삼각파 샘플된 것 사진)
        
        - 만약 더 작으면, 반복되는 모양에 겹치는 게 생겨서 복구 불가능하다 (Aliasing distortion)
        - 바꿔 말하면, 샘플링 속도에 비해 freq가 1/2 이상인 신호는 복구 불가능하다.
            - (직관적) 어떤 사인파를 복구해야 하는데, 사인파 주기의 절반에서 샘플링을 하면, 사인파 정보를 필요충분하게 알 수 있다. ( 최대 - 최소 계속 반복 )
        - 샘플링 속도?
            - 빠름 : 신호 왜곡 적어짐, but 데이터가 너무 커진다
            - 느림 : 신호 왜곡 생김 but 데이터 양 감소
            - 사람 귀 주파수는 20~22000HZ → **44,100HZ** (2배보다 조금 더 크다)
            - 사람 말소리는 ~8KHZ → 16KHZ sampling
            
        - Antialiasing filter
            - Signal 관점에서, sample rate에 맞춰서 대역폭을 제한해주는 필터 ( $f_s/2$ )
            - 시그널에 이 필터를 적용한 뒤 sampling device로 보냄
        - Oversampled
            - 예를들어, Nyquist rate의 3배로 샘플을 하면, 사실 3k번째 샘플만 있어도 okay.
            - 그런데도 샘플링주기를 늘리는 이유 : 현실적으로, Antialiasing filter가 이상적으로 [ $-f_m,f_m$ ] 으로 잘라주지 못함 : 간격을 충분히 두어서 distortion 제거
            - 
        
    - Fundamental Frequency
        
        <aside>
        💡 **요약**
        
        </aside>
        
        - 음성은 여러 주파수에 해당하는 많은 함수들이 합성되어 있어 매우 복잡함
        → 작은 부분으로 쪼개서(시간적) 분석
        - 잡음이란?
            - 소리 = 주기적인 특성 + 비주기적인 특성
            - 잡음 = 비주기적인 특성
        - 주기적인 특성을 가진 소리(ex. 유성음)
            - time domain에서 : 주기성
            - frequency domain에서 : formant frequency 나타남( 강조되는 특정 주파수)
                - 사운드를 대표하는 특징이 되며, 우리가 인지하는 소리 높낮이에 영향
                - formant 1(F1), formant 2(F2) ….
                - formant에 대응하는 주기 : pitch
            - 이 formant들의 분포로 유성음 구별 가능
    - Spectrogram
        
        <aside>
        💡 time 축, frequency 축을 가지는 이미지로, 시간에 따른 주파수 분포를 나타내 줌
        
        기존 그래프들(time - amplitude, frequency - amplitude)
        
        </aside>
        
        - 어떤 음성을, 각 **시간마다의 주파수 분포**? 를 나타내주는 Image
            - X축: Time
            - Y축: Frequency
            - (time, frequency)에 대응되는 값(밝기) : 진폭
        - Band Pass filter ( -x ~ x 사이 주파수 영역 ) 활용해서, 그 영역 내의 주파수 나타내줌
        - 시간 축을 고정시키면, 각 Frequency 별로, 센 값은 어둡게 나타남 (흑백에서)
            - 어두운 부분 : formant
            - formant만을 추출해서 time, frequency 축에 나타낸, 간소화된 형태의
            formant frequency representation spectrogram도 가능.
        - **시간** 변화에 따른 **주파수** 성분들 변화 분석 가능
        
    - 요약
        - 아날로그 신호 : time - amplitude continuous signal
        - Analog → Digital : time -amplitude discrete signal ( by quantization, sampling )
            - Nyquist Sampling, $\omega_s > 2\omega_b$
        - (Digital) Time domain signal → frequency domain signal ( by DFT ) ← 수학적으로 좀 더 분석 필요
            - formant
        - Spectrogram : time series + frequency distribution +

- 조음 / 음성합성모델
    - 기류의 조달과정 → 조절과정 → 변형과정
    - 발동(폐) → 발성(성대) → 조음(성도, vocal tract)
    - 그냥 공기 흐름 → 말소리의 기본 성격 가짐 → 특정 음가 가짐
    - 성대는 발동, 발성, 조음 모두 관여
        - pitch 결정
    
    - 각종 음가들은 조음기관의 이런저런 작용에 따라 분류됨
        - 자음, 모음
            - 자음: 조음기관 방해 받음 → 모음에 비해 signal이 작다.
        - 자음 분류
            - 조음방법, 조음자리, 기의 유무, 긴장의 유무
        - 모음 분류 (+ 단순모음, 이중모음)
            - 혀 위치, 높이, 입술모양
            - 영어에서는 추가로 Nasalized or Unnaslized(비음화) 기준이 있는듯( ɑ̃ )
            
    - 조음기관들을 모델화
        - 발동(sound source) → 발성, 조음(filter, 성대~입술) → sound output
        
    - 유성음 / 무성음
        - 유성음(voiced sound)의 특징 : formant, 주기성, …
            - 폐로부터 유사 **주기적**인 기류 공급
            - formant로 유성음 특성 규정 가능
            - fundamental frequency(F0)
        - 무성음(unvoiced sound)의 특징 : noisy, 비주기성, …
        
    - 음성합성 모델 (순서대로) : 실제 사람이 소리 내는 방법 모방
        - 유성음 → Impulse generator, pitch period가 parameter
        - 무성음 → white noise generator
        - 이 두개 switch로 결정
        - Gain estimate : 소리 크기 parameter
        - Filter
            - Vocal tract
            - Lip radiation
    
    - 음성인식 모델 : 위 모델을 활용하여, 역으로 parameter 추적해서 인식
        - [Voice and Speech Processing 요약](https://www.notion.so/Voice-and-Speech-Processing-d1896c011e5847db9f3ef127297e8031)
- 음성인식 ( 데이터 )
    1. sound data → words sequence
        - 음성 : silence + signal + silence ← 끝점검출
        - 잘린 음성 → (특징추출) → input vector
        - 음성인식에서는, 프레임들을 조금씩 겹친 걸 가지고 input vector sequence를 만든다
            - 겹치지 않고 자르면 그 잘리는 경계선에서 어색한 학습이 될 수 있음
        - input vector → (model) → 단어열/문장
        
    2. ~~Words sequence → Meaning (자연어처리)~~
    3. ~~대화처리.~~
    - 
- Hidden Markov / 음성인식 
* Hidden Markov Model : [https://hyunlee103.tistory.com/52](https://hyunlee103.tistory.com/52)
    
    ### 1. Hidden Markov Model
    
    - 베이즈 정리 → 선후관계 바뀌어 측정 불가한 조건부확률 추정 가능
        - $P(A|B) = {P(B|A)P(A)}/P(B)$
    - 음성인식 목표 : $P(w_1w_2w_3…w_n)$ 구하기
        - $w_1w_2w_3…w_n$ : 단어열
        - 확률 젤 큰 것을 답으로 골라야 함
    - Markov Assumption
        - i번째 단어 확률은 바로 직전 i-1번째 단어에만 영향 받음
        - $P(w_i|w_1w_2w_3 … w_{i-1}) \approx P(w_i|w_{i-1})$
        - $\therefore        P(w_1w_2…w_n) = \prod_{i=1}^{n}  p(w_i|w_{i-1})$
        - Markov Chain 생긴 거 보면, 확률이 바로 앞 state에만 영향을 받음을 알 수 있다.
    
    [Hidden Markov Model](https://www.notion.so/Hidden-Markov-Model-64bc029563434415bd65eeaf8b564bbd)
    
    - 음성인식에서,
        - input : O (acoustic input, 이런저런 전처리 통해서 얻어진 벡터) // 이 음성이
        - output : $\hat{W}$ (예상 단어 열)  // 무슨 단어열인지?
        - $**\hat{W}$ = $Argmax_{W{\in}L}P(O|W)P(W)$** ← by 베이즈정리, P(O)는 그냥 1
    - $P(O|W)$ : Acoustic model 확률
    - $P(W)$ : Language model 확률
    
    ### 2. 실제 feature 추출 ( O 만들기 ) : MFCC
    
     [https://ratsgo.github.io/speechbook/docs/fe/mfcc](https://ratsgo.github.io/speechbook/docs/fe/mfcc) 
    
     [https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) 
    
    1) pre - emphasis : 고주파 증폭 ← resolution 향상
    
    - 주파수 영역별로, 에너지 분포 고르게 해줌 (특히 작은 고주파 증폭)
    - 푸리에변환시의 Numerical Problem 예방 ( zero issue )
    - Resolution 향상
    - (practical) $\alpha$ = 0.95 or 0.97
    
    2) windowing : ‘겹친’ 여러 frame으로 자르기 + 경계 smoothing
    
    - 불연속성 issue 해결
    - 25ms 정도
        - 짧게 잡음으로서, stationary signal 가정 가능. // time에 영향 안받는거(교재참고)
    - 10ms 정도는 겹치게
    - Hamming Window(이걸 주로 많이씀) : 경계값 문제
        - [https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet#windowing--hamming-window](https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet#windowing--hamming-window)
        - Gibbs phenomenon
        - 다른 윈도우도 쓰긴 하는 거 같다. Hanning Window .. ?
    
    3) DFT ( to spectrum )
    
    - 정확히는, 프레임들로 잘려진 곳에다가 DFT 해서 frequency로 만든 게 스펙트럼이다.
    - 실제로는 **FFT** ( Fast Fourier Transfrom 을 사용 )
        - frequency bin 몇개 ?  : N = 256 or 512 …
        - (trivial) np.fft.rfft ( y축 대칭이니까 절반만 계산 )
        - 결과는 complex이다. phase 안쓰고 magnitude만 쓰므로 abs()
        - Power Spectrum 만들어서 쓰기. // 이거 정확히 뭔지 확인
    
    4) mel filter bank
    
    - Mel Spectrum : 사람 말소리 민감한 freq 영역 → 세밀하게, 나머지 → 덜 세밀하게 보는 filter 적용. ( 사람 귀 선형적 x, 저주파에서 더 민감한 거 반영 )
    - Mel Scale
    - log - Mel Spectrum : to log scale
    
    5) $DFT^{-1}$ ( Cepstrum )
    
    - freq domain을 다시 time domain으로, why ?
        - Mel spectrum은 피쳐들 간의 correlation 필연적으로 존재 ( 필터 계산 생각해보면, 근처 주파수대 애들 선형조합이 된다. )
        - time domain으로 바꿔서, freq 변수 간 상관관계 해소해 준다는 개념
        - 그 맨 처음 샘플같은 time domain은 아니고, 그냥 축이 시간 축이 되는 아예 다른 데이터(인 것 같다) → **Quefre**ncy
    - 정확히는, 역 코사인 변환을 수행함 ( 실수 파트만 다룸 )
        - Consider, $e^{i\theta}=isin\theta + cos\theta$
    - $F_0$을 구할 수 있다고 함
        - Cepstrum에서 튀어 나온 부분으로 나타남
        - remind) 이 때, X축은 시간이다 ( Freq 역수 )
    - Harmony peak($F_0$ 의 정수배)과  Formant 분리
        - Harmony Peak ↔ Spectral Envelope
        - 즉, 기본주파수 ↔ Formant
        - (간단한 원리) log가 적용됐었으므로, logab = loga + logb 이런 느낌으로 분리된다. (교재 P.203)
        - 
    
    6) 이 Cepstrum을 가지고 다음과 같은 39개 feature vector을 input O 로 최종적으로 활용 (MFCC) ← 추가학습필요
    
    - 5)까지의 결론 : Cepstrum data는, Fundamental Freq, Formant 등의 정보들이 내재되어 있다. 그 밖에 log mel spectrum으로부터 왔으므로, recognition에 관한 이런저런 feature가 많이 들어있을 것이다.
    - 12 Cepstral Coefficients + 1 energy Coefficient
    - “ $\Delta$
    - “ $\Delta\Delta$
    - Lift, Mean Norm 등 후처리.
    - 최근에는 Neural Network로 feature 추출하기도 한다고 함. 근데 여전히 이게 좋은 feature.
    - 근데 버리는 정보가 좀 많아서 최근에는 또 그냥 멜스펙트럼을 쓰기도 한다고 함