import os
import json
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv
from openai import OpenAI
import tweepy

try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False

load_dotenv()

client_oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_tw = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"), wait_on_rate_limit=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MIN_FOLLOWERS = 5000
MIN_TWEETS_2W = 6

def generate_keywords(topic="US financial markets", n=3):
    resp = client_oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Return a JSON array of keywords."},
            {"role": "user", "content": f"Give {n} short keywords for: {topic}"}
        ]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return ["US stock market", "Wall Street", "S&P 500"]

def search_authors(keywords, days=7, per_kw=10):
    start_time = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    authors = {}
    for kw in keywords:
        try:
            resp = client_tw.search_recent_tweets(
                query=f'{kw} -is:retweet lang:en',
                max_results=min(per_kw, 100),
                start_time=start_time,
                expansions=["author_id"],
                user_fields=["username","public_metrics"],
            )
            if resp.includes and "users" in resp.includes:
                for u in resp.includes["users"]:
                    authors[str(u.id)] = {
                        "username": u.username,
                        "followers": u.public_metrics.get("followers_count", 0)
                    }
        except Exception as e:
            print("warn:", e)
    return authors

def count_tweets(uid, days=7):
    start_time = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        resp = client_tw.get_users_tweets(id=uid, start_time=start_time, max_results=100)
        return len(resp.data) if resp and resp.data else 0
    except:
        return 0

def run_pipeline(topic="US financial markets"):
    kws = generate_keywords(topic)
    authors = search_authors(kws)
    results = []
    for uid, info in authors.items():
        t = count_tweets(uid)
        if info["followers"] >= MIN_FOLLOWERS and t >= MIN_TWEETS_2W:
            results.append({
                "profile_url": f"https://x.com/{info['username']}",
                "username": info["username"],
                "followers": info["followers"],
                "tweets_last_2_weeks": t,
                "avg_posts_per_week": round(t/2,2)
            })
    with open("results.json","w",encoding="utf-8") as f:
        json.dump({"topic":topic,"results":results},f,indent=2)
    print(f"saved {len(results)} users to results.json")
    return results

def run_crewai(topic="US financial markets"):
    if not CREWAI_AVAILABLE:
        return run_pipeline(topic)

    gen_keywords_agent = Agent(
        role="Keyword Generator",
        goal="Generate keywords for searching creators about financial markets",
        backstory="Expert at picking concise keywords.",
        llm=OPENAI_MODEL,
        verbose=True
    )

    def search_and_filter(ctx):
        kws = generate_keywords(topic)
        authors = search_authors(kws)
        results = []
        for uid, info in authors.items():
            t = count_tweets(uid)
            if info["followers"] >= MIN_FOLLOWERS and t >= MIN_TWEETS_2W:
                results.append({
                    "profile_url": f"https://x.com/{info['username']}",
                    "username": info["username"],
                    "followers": info["followers"],
                    "tweets_last_2_weeks": t,
                    "avg_posts_per_week": round(t/2,2)
                })
        return results

    t1 = Task(
        description=f"Run search & filter pipeline for {topic}",
        agent=gen_keywords_agent,
        expected_output="Final JSON list of users",
        func=search_and_filter
    )

    crew = Crew(
        agents=[gen_keywords_agent],
        tasks=[t1],
        verbose=True
    )

    try:
        result = crew.kickoff()
        if hasattr(result, "raw"):
            result_data = result.raw
        elif hasattr(result, "to_dict"):
            result_data = result.to_dict()
        elif hasattr(result, "json"):
            result_data = result.json
        else:
            result_data = result
    except Exception as e:
        print("CrewAI flow failed:", e)
        result_data = run_pipeline(topic)

    with open("results_crewai.json", "w", encoding="utf-8") as f:
        json.dump({"topic": topic, "results": result_data}, f, indent=2)

    print(f"CrewAI flow finished, saved {len(result_data)} users to results_crewai.json")
    return result_data

if __name__=="__main__":
    topic="US financial markets"
    if CREWAI_AVAILABLE:
        run_crewai(topic)
    else:
        run_pipeline(topic)
