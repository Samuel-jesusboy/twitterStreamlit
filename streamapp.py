import streamlit as st
from twitter_sent import *


# Define a CSS class for custom styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://source.unsplash.com/cfKC0UOZHJo");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
def main():
    st.title("Twitter Sentiment Analysis App")
    st.markdown("## Welcome to Twitter Insights!\n\nThis is a simple interface to generate quick insights into a Twitter user's activity. With just a Twitter username, you can effortlessly generate comprehensive reports, assess sentiment, and estimate networking scores. Dive into the world of social media analytics and discover the power of data-driven decisions.")

    # Input widget for Twitter username
    username = st.text_input("Enter a valid Twitter username (Example: elonmusk):")

    # Buttons for different actions
    if st.button("Generate a Sentiment_report"):
        if username:
            # Create a text element for displaying the output
            output_text = st.empty()
            
            # Call the sentiment_report function and capture the result
            result = finalReport(sentiment(tweetTodf(get_tweet(username))))
            score = sentiment_score(sentiment(tweetTodf(get_tweet(username))))

            # Display the result in Streamlit
            with st.spinner("Running Sentiment Report and scoring..."):
                output_text.text(f"The sentiment score of this user is: {score}")
                # output_text.text("Sentiment Report:")
                # output_text.text(result)

                # Display the saved pie chart images
                st.header("Sentiment Report")
                st.markdown("* The final data sample")
                st.write(result)

                st.markdown("* visualization of the sentiment result")
                st.image("star_pie_chart.png", use_column_width=True, caption="Percentage of Total tweets by star ratings")
                st.image("sentiment_pie_chart.png", use_column_width=True, caption="Percentage of total tweets by sentiment")

                st.markdown("* Most occurring words in each sentiment category: ")
                # Display the wordcloud
                st.image("wordcloud_negative.png", use_column_width=True, caption="words occuring the most in negative tweets")
                st.image("wordcloud_neutral.png", use_column_width=True, caption="words occuring the most in neutral tweets")
                st.image("wordcloud_positive.png", use_column_width=True, caption="words occuring the most in positive tweets")

        else:
            st.warning("Please enter a valid Twitter username.")

    # if st.button("Sentiment_score"):
    #     if username:
    #         # Create a text element for displaying the output
    #         output_text = st.empty()

    #         # Call the sentiment_score function and capture the result
    #         score = sentiment_score(sentiment(tweetTodf(get_tweet(username))))

    #         # Display the result in Streamlit
    #         with st.spinner("Running Sentiment Score..."):
    #             output_text.text(f"The sentiment score of this user is: {score}")
    #     else:
    #         st.warning("Please enter a valid Twitter username.")

    if st.button("Generate a networking Score"):
        if username:
            # Create a text element for displaying the output
            output_text = st.empty()

            #call the networking score function and capture the result
            net_score = generate_networking_score(username)

            # Display the result in Streamlit
            with st.spinner("Running Networking scoring..."):
                output_text.text(f"The Networking score of this user is: {net_score}")
        else:
            st.warning("Please enter a Twitter username.")

if __name__ == "__main__":
    main()
