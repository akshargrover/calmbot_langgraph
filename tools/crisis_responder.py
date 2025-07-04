def crisis_responder(state):
    return {
        **state,
        "crisis_response": (
            "I'm really sorry you're feeling this way. You're not alone — there are people who care about you and want to help.\n\n"
            "💙 Please reach out to someone you trust or contact a mental health professional.\n\n"
            "**If you're in immediate danger**, please call emergency services or reach out to a suicide prevention hotline:\n\n"
            "📞 India Helpline: 9152987821 (iCall)\n"
            "🌐 International: https://findahelpline.com\n\n"
            "Remember, you're not alone. There are people who care about you and want to help."
        )
    }
