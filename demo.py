import gradio as gr
#https://www.sbert.net/
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


def run(query, start=2015, end=2023, genre = None):
    #pre-trained 모델 사용. https://github.com/jhgan00/ko-sentence-transformers 참고.
    embedder1 = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda')
    embedder2 = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device='cuda')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    songs = pd.read_csv("csvs/songs.csv", index_col=0)

    #load embeddings -이미 만들어진 임베딩을 불러올 때 사용
    corpus_embeddings = np.load('csvs/lyrics_embeddings.npy')
    corpus_embeddings = torch.as_tensor(corpus_embeddings, device=device)

    top_k = 1000
    top_real = 30 # 실제 출력할 개수
    query_embedding1 = embedder1.encode(query, convert_to_tensor=True)
    query_embedding2 = embedder2.encode(query, convert_to_tensor=True)
    query_embedding = (query_embedding1 + query_embedding2) / 2
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0].cpu()

    ret = []
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    print("\n\n======================\n")
    print("Query:", query)
    print(f"Top {str(top_real)} most similar songs")
    i = 0
    for idx in top_results[0:top_k]:
        try:
            song_year = songs["date"].iloc[int(idx)][:4]
            # 필터링 조건
            if (len(songs['lyrics'].iloc[int(idx)]) > 50 and song_year != '-' and start <= int(song_year) <= end) and genre in songs['genre'].iloc[int(idx)]:
                i += 1
                print(songs['song_name'].iloc[int(idx)])
                ret.append([songs['song_name'].iloc[int(idx)], songs['artist'].iloc[int(idx)],
                   int(cos_scores[idx].item()*10000)/10000,songs['genre'].iloc[int(idx)], songs['date'].iloc[int(idx)], "https://www.melon.com/song/detail.htm?songId="+str(songs.iloc[[int(idx)]].index.values.item())])
                print(ret)
                if i == top_real:
                    break
        except:
            continue
    print(ret)
    return pd.DataFrame(ret, columns=["곡명", "가수", "유사도", "장르", "발매일자", "링크"])


with gr.Blocks() as demo:
    with gr.Row():
        query = gr.Textbox(label="문장 입력하기")
        with gr.Column():
            with gr.Row():
                start = gr.Slider(1970, 2023, value=2015, step=1, label="부터")
                end = gr.Slider(1970, 2023, value=2023, step=1, label="까지")
            genre = gr.Textbox(label="장르")
            btn = gr.Button(value="Submit")

    txt_3 = gr.DataFrame(headers=["곡명", "가수", "유사도", "장르", "발매일자", "링크"], label="출력")
    btn.click(fn=run, inputs=[query, start, end, genre], outputs=txt_3)

    gr.Markdown("예시")
    gr.Examples(
        ["""일부러 몇 발자국 물러나
내가 없이 혼자 걷는 널 바라본다
옆자리 허전한 너의 풍경
흑백 거리 가운데 넌 뒤돌아본다
그때 알게 되었어
난 널 떠날 수 없단 걸
우리 사이에 그 어떤 힘든 일도
이별보단 버틸 수 있는 것들이었죠
어떻게 이별까지 사랑하겠어
널 사랑하는 거지
사랑이라는 이유로 서로를 포기하고
찢어질 것 같이 아파할 수 없어 난
두세 번 더 길을 돌아갈까
적막 짙은 도로 위에 걸음을 포갠다
아무 말 없는 대화 나누며
주마등이 길을 비춘 먼 곳을 본다
그때 알게 되었어
난 더 갈 수 없단 걸
한 발 한 발 이별에 가까워질수록
너와 맞잡은 손이 사라지는 것 같죠
어떻게 이별까지 사랑하겠어
널 사랑하는 거지
사랑이라는 이유로 서로를 포기하고
찢어질 것같이 아파할 수 없어 난 no oh oh
어떻게 내가 어떻게 너를
이후에 우리 바다처럼 깊은 사랑이
다 마를 때까지 기다리는 게 이별일 텐데
어떻게 내가 어떻게 너를
이후에 우리 바다처럼 깊은 사랑이
다 마를 때까지 기다리는 게 이별일 텐데""",
"""Stay in the middle
Like you a little
Don't want no riddle
말해줘 say it back, oh, say it ditto
아침은 너무 멀어 so say it ditto
훌쩍 커버렸어
함께한 기억처럼
널 보는 내 마음은
어느새 여름 지나 가을
기다렸지 all this time
Do you want somebody
Like I want somebody?
날 보고 웃었지만
Do you think about me now? Yeah
All the time, yeah, all the time
I got no time to lose
내 길었던 하루, 난 보고 싶어
Ra-ta-ta-ta 울린 심장 (Ra-ta-ta-ta)
I got nothing to lose
널 좋아한다고 ooh-whoa, ooh-whoa, ooh-whoa
Ra-ta-ta-ta 울린 심장 (Ra-ta-ta-ta)
But I don't want to
Stay in the middle
Like you a little
Don't want no riddle
말해줘 say it back, oh, say it ditto
아침은 너무 멀어 so say it ditto
I don't want to walk in this 미로
다 아는 건 아니어도
바라던 대로 말해줘 say it back
Oh, say it ditto
I want you so, want you, so say it ditto
Not just anybody
너를 상상했지
항상 닿아있던
처음 느낌 그대로 난
기다렸지 all this time
I got nothing to lose
널 좋아한다고 ooh-whoa, ooh-whoa, ooh-whoa
Ra-ta-ta-ta 울린 심장 (Ra-ta-ta-ta)
But I don't want to
Stay in the middle
Like you a little
Don't want no riddle
말해줘 say it back, oh, say it ditto
아침은 너무 멀어 so say it ditto
I don't want to walk in this 미로
다 아는 건 아니어도
바라던 대로 말해줘 say it back
Oh, say it ditto
I want you so, want you, so say it ditto""",
         """널 품기 전 알지 못했다
내 머문 세상 이토록
찬란한 것을
작은 숨결로 닿은 사람
겁 없이 나를 불러준 사랑
몹시도 좋았다
너를 지켜보고 설레고
우습게 질투도 했던
평범한 모든 순간들이
캄캄한 영원
그 오랜 기다림 속으로
햇살처럼 니가 내렸다
널 놓기 전 알지 못했다
내 머문 세상 이토록
쓸쓸한 것을
고운 꽃이 피고 진 이 곳
다시는 없을 너라는 계절
욕심이 생겼다
너와 함께 살고 늙어가
주름진 손을 맞잡고
내 삶은 따뜻했었다고
단 한번 축복
그 짧은 마주침이 지나
빗물처럼 너는 울었다
한번쯤은 행복하고
싶었던 바람
너까지 울게 만들었을까
모두, 잊고 살아가라
내가 널, 찾을 테니
니 숨결, 다시 나를 부를 때
잊지 않겠다
너를 지켜보고 설레고
우습게 질투도 했던
니가 준 모든 순간들을
언젠가 만날
우리 가장 행복할 그날
첫눈처럼 내가 가겠다
너에게 내가 가겠다""",
         """Umm 내가 슬플 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우린 (동글동글)
인생은 회전목마
우린 매일 달려가
언제쯤 끝나 난 잘 몰라 (huh, huh, huh)
어머 (어머), 벌써 (벌써) 정신없이 달려왔어 (왔어)
Speed up (speed up) 어제로 돌아가는 시곌 보다가
어려워 (어려워) 어른이 되어가는 과정이 uh huh
On the road, twenty four 시간이 아까워 uh huh
Big noise, everything brand new
어렸을 때처럼 바뀌지 않는 걸
찾아 나섰단 말야 왜냐면 그때가 더 좋았어 난
So let me go back
타임머신 타고 I'll go back
승호가 좋았을 때처럼만
내가 슬플 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우리
인생은 회전목마
우린 매일 달려가
언제쯤 끝나 난 잘 몰라
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마 ayy
어머 (어머) 벌써 (벌써) 정신없이 달려왔어 (왔어)
Speed up (speed up) 어제로 돌아가는 시곌 보다가
청춘까지 뺏은 현재 (현재)
탓할 곳은 어디 없네
Twenty two 세에게 너무 큰 벽
그게 말로 하고 싶어도 어려웠어
가끔은 어렸을 때로 돌아가
불가능하단 건 나도 잘 알아
그 순간만 고칠 수 있다면
지금의 나는 더 나았을까
달려가는 미터기 돈은 올라가
기사님과 어색하게 눈이 맞아
창문을 열어보지만 기분은 좋아지지 않아
그래서 손을 밖으로 쭉 뻗어 쭉 뻗어
흔들리는 택시는 어느새
목적지에 도달했다고 해
방 하나 있는 내 집 안의 손에 있던 짐들은
내가 힘들 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우리
인생은 회전목마
우린 계속 달려가
언제쯤 끝날지 잘 몰라
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
우 인생은 회전목마
I'm on a TV show
You would never even know
사실 얼마나 많이 불안했는지
정신없이 돌아서
어딜 봐야 할지 모르겠어
들리나요 여길 보란 말이
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마""",
         """
         무슨 이유로 태어나
어디서부터 왔는지
오랜 시간을 돌아와
널 만나게 됐어

의도치 않은 사고와
우연했던 먼지덩어린
별의 조각이 되어서
여기 온 거겠지

던질수록 커지는 질문에
대답해야 해

돌아갈 수 있다 해도
사랑해 버린 모든 건
이 별에 살아 숨을 쉬어
난 떠날 수 없어

태어난 곳이 아니어도
고르지 못했다고 해도
나를 실수했다 해도
이 별이 마음에 들어

까만 하늘 반짝이는
거기선 내가 보일까
어느 시간에 살아도
또 만나러 올게

그리워지면 두 눈을 감고
바라봐야 해

돌아갈 수 있다 해도
사랑해 버린 모든 건
이 별에 살아 숨을 쉬어
난 떠날 수 없어

태어난 곳이 아니어도
고르지 못했다고 해도
내가 실수였다 해도
이 별이 마음에 들어

언젠가 만날 그날을
조금만 기다려줄래
영원할 수 없는 여길
더 사랑해 볼게

돌아갈 수 있다 해도
사랑해 버린 모든 건
이 별에 살아 숨을 쉬어
난 떠날 수 없어

태어난 곳이 아니어도
고르지 못했다고 해도
내가 실수였다 해도
이 별이 마음에 들어

낮은 바람의 속삭임
초록빛 노랫소리와
너를 닮은 사람들과
이 별이 마음에 들어""",
         """Look at you 넌 못 감당해 날
Ya took off hook
기분은 Coke like brrr
Look at my toe 나의 Ex 이름 Tattoo
I got to drink up now 네가 싫다 해도 좋아

Why are you cranky, boy?
뭘 그리 찡그려 너
Do you want a blond barbie doll?
It’s not here, I’m not a doll

미친 연이라 말해 What’s the loss to me ya
사정없이 까보라고 You’ll lose to me ya
사랑 그깟 거 따위 내 몸에 상처 하나도 어림없지
너의 썩은 내 나는 향수나 뿌릴 바엔

Ye I’m a Tomboy (Umm ah umm)
Ye I’ll be the Tomboy (Umm ah)
This is my attitude
Ye I’ll be the Tomboy

I don’t wanna play this ping pong
I would rather film a Tik Tok
Your mom raised you as a prince
But this is queendom, right?
I like dancing, I love ma friends
Sometimes we swear without cigarettes
I like to eh on drinking whiskey
I won’t change it, what the hell?

미친 척이라 말해 What’s the loss to me ya
사정없이 씹으라고 You’re lost to me ya
사랑 그깟 거 따위 내 눈에 눈물 한 방울 어림없지
너의 하찮은 말에 미소나 지을 바엔

Ye I’m a Tomboy (Umm ah umm)
Ye I’ll be the Tomboy (Umm ah)
This is my attitude
Ye I’ll be the Tomboy

Said you get it?
You get the song right, you’ll get what I mean “Tomboy”

La la la la la la la la la
La la la la la la la la la

La la la la la la la la la
La la la la la la la la la
(Three, two, one)

It’s neither man nor woman
Man nor woman
It’s neither man nor woman
(Just me I-DLE)

It’s neither man nor woman
Man nor woman
It’s neither man nor woman
(Just me loving Tomboy)"""],
        [query],
        txt_3,
        run,
    )


    demo.launch(share=False)



