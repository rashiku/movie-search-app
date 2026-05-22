def fetch_tmdb_poster(movie):

    title = movie["title"]

    release_date = movie["release_date"]

    # ---------------------------------------------------
    # CLEAN YEAR
    # ---------------------------------------------------

    year = ""

    if pd.notna(release_date):

        year = str(
            release_date
        )[:4]

    # ---------------------------------------------------
    # NORMALIZED CACHE KEY
    # ---------------------------------------------------

    cache_key = (
        title.strip().lower()
        + "_"
        + year
    )

    # ---------------------------------------------------
    # SKIP CACHED
    # ---------------------------------------------------

    if cache_key in poster_cache:

        return (
            cache_key,
            poster_cache[cache_key],
            True
        )

    # ---------------------------------------------------
    # FETCH FROM TMDB
    # ---------------------------------------------------

    try:

        url = (
            "https://api.themoviedb.org/"
            "3/search/movie"
        )

        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "year": year
        }

        response = session.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        results = data.get(
            "results",
            []
        )

        poster_url = None

        if results:

            poster_path = results[
                0
            ].get("poster_path")

            if poster_path:

                poster_url = (
                    POSTER_BASE_URL
                    + poster_path
                )

        return (
            cache_key,
            poster_url,
            False
        )

    except Exception as e:

        print(
            f"Error fetching "
            f"{title}: {e}"
        )

        return (
            cache_key,
            None,
            False
        )