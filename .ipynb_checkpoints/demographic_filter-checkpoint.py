import pandas
class RecommendationSystem:
  
    def __init__(self,data):
        self.df = pandas.read_csv(data)
    
    def recommend(self,genre=None,duration=None,year=None,topk=10):
        df = self.df.copy()
        df = self.demographic_filter(df,genre=genre,duration=duration,year=year)
        df = self.compute_imdb_score(df)
        
        result = df.loc[:,"title":"release_year"]
        result = result.sort_values("vote_average",ascending=False)
        result = result.head(topk)
        return result
    
    @staticmethod
    def demographic_filter(df,genre=None, duration=None,year=None):
        df = df.copy()
        
        if genre is not None:
            df = df[df[genre].all(axis=1)]
        if duration is not None:
            df = df[df.runtime.between(duration[0],duration[1])]
        if year is not None:
            df = df[df.release_year.between(year[0],year[1])]
            
        return df
        
    @staticmethod
    def compute_imdb_score(df, q=0.8):
        df = df.copy()
        m = df.vote_count.quantile(q)
        C = (df.vote_average * df.vote_count).sum() / df.vote_count.sum()
        
        df = df[df.vote_count >= m]
        df["score"] = df.apply(lambda x: (x.vote_average * x.vote_count + C*m)/ (x.vote_count + m), axis=1)
        return df
    