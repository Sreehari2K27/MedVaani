"""Deterministic, offline translation of patient-facing assistant text.

The voice endpoint keeps the English ``response_text`` for the on-screen UI and
adds a separate spoken form for TTS. This module produces that spoken form.

Scope and honesty rules:
- Only the fixed patient-facing templates (greetings, acknowledgements,
  questions, confirmations, safety messages and the closing line) are
  translated. Structured values such as names, phone numbers, product names
  and dates are never translated; they are interpolated unchanged.
- Translation is all-or-nothing per response: if any sentence has no known
  translation, the whole response is returned in English with
  ``translated=False``. The endpoint then speaks English rather than mixing
  languages or claiming a language it cannot actually speak.
- Only Hindi (``hi-IN``) is populated today. The catalog is intentionally a
  plain dict so other languages can be added without touching callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern

from .catalog import DEFAULT_LANGUAGE, supports_language

_ENGLISH = DEFAULT_LANGUAGE

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TranslationResult:
    """Outcome of translating one assistant response."""

    text: str
    translated: bool
    language: str
    source_language: str = _ENGLISH


# Exact sentence -> Hindi. Keys include the trailing punctuation so they match
# the sentences produced by the workflow verbatim.
_HINDI_SENTENCES: dict[str, str] = {
    "Thanks for that.": "इसके लिए धन्यवाद।",
    "Got it, thank you.": "समझ गया, धन्यवाद।",
    "Okay, thank you.": "ठीक है, धन्यवाद।",
    "Thanks for the details.": "जानकारी देने के लिए धन्यवाद।",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "मुझे बताने के लिए धन्यवाद, और मुझे खेद है कि आपको यह सब झेलना पड़ रहा है।"
    ),
    "I'll take a few details so we can record it safely.": (
        "मैं कुछ जानकारी लूँगा ताकि हम इसे सुरक्षित रूप से दर्ज कर सकें।"
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "महत्वपूर्ण: बताए गए दुष्प्रभावों के मूल्यांकन और उपचार के लिए "
        "कृपया किसी योग्य चिकित्सक या स्वास्थ्य विशेषज्ञ से परामर्श करें।"
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "चिकित्सकीय सलाह के बिना स्वयं निदान न करें और अपनी दवा न बदलें।"
    ),
    "Which medicine or product do you think may have caused this?": (
        "आपको क्या लगता है कि किस दवा या उत्पाद से यह हुआ हो सकता है?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "क्या आप बता सकते हैं कि आपने कौन-सा दुष्प्रभाव या प्रतिक्रिया देखी?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "क्या आप वही व्यक्ति हैं जिसे यह हुआ, या आप किसी और की ओर से यह बता रहे हैं?"
    ),
    "What is the patient's full name?": "रोगी का पूरा नाम क्या है?",
    "May I have your first and last name?": "क्या मैं आपका पूरा नाम जान सकता हूँ?",
    "How old is the patient, or what is their date of birth?": (
        "रोगी की उम्र क्या है, या उनकी जन्मतिथि क्या है?"
    ),
    "How old are you, or what is your date of birth?": (
        "आपकी उम्र क्या है, या आपकी जन्मतिथि क्या है?"
    ),
    "And what is your name?": "और आपका नाम क्या है?",
    "If we need a little more information later, is it alright if we contact you?": (
        "अगर बाद में हमें थोड़ी और जानकारी चाहिए, तो क्या हम आपसे संपर्क कर सकते हैं?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "आप फ़ोन से संपर्क करना चाहेंगे या ईमेल से?"
    ),
    "What phone number would you like us to use?": "आप कौन-सा फ़ोन नंबर देना चाहेंगे?",
    "What email address would you like us to use?": "आप कौन-सा ईमेल पता देना चाहेंगे?",
    "When did the side effect first start?": "दुष्प्रभाव पहली बार कब शुरू हुआ?",
    "Would you consider this problem serious?": "क्या आप इस समस्या को गंभीर मानते हैं?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "अब रोगी कैसा महसूस कर रहा है - क्या लक्षण बेहतर हुए हैं, वैसे ही हैं, या और बिगड़ गए हैं?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "अब आप कैसा महसूस कर रहे हैं - क्या आपके लक्षण बेहतर हुए हैं, वैसे ही हैं, या और बिगड़ गए हैं?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "क्षमा करें, मैं समझ नहीं पाया - अगर यह सही है तो हाँ कहिए, या बदलना है तो ना कहिए।"
    ),
    "Is that correct?": "क्या यह सही है?",
    "Thank you - I've recorded everything we need for now.": (
        "धन्यवाद - मैंने अभी के लिए वह सब दर्ज कर लिया है जो हमें चाहिए था।"
    ),
    "A member of our team will follow up if needed.": (
        "अगर ज़रूरत हुई तो हमारी टीम का कोई सदस्य आपसे संपर्क करेगा।"
    ),
    "Take care.": "अपना ध्यान रखें।",
    "Thank you.": "धन्यवाद।",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "समाप्त करने से पहले, क्या आप इस बारे में कुछ और बताना चाहेंगे कि क्या हुआ?"
    ),
    "Thank you for sharing that.": "यह बताने के लिए धन्यवाद।",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "अगर यह कोई मेडिकल आपातकाल है, तो कृपया तुरंत चिकित्सकीय सहायता लें "
        "या अपनी स्थानीय आपातकालीन सेवा को कॉल करें।"
    ),
    "This appears potentially urgent.": "यह संभावित रूप से आपातकालीन लगता है।",
    "A healthcare professional should be contacted immediately.": (
        "तुरंत किसी स्वास्थ्य विशेषज्ञ से संपर्क करना चाहिए।"
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "मैं व्यक्तिगत खुराक, उपचार सिफ़ारिशें या दवा में बदलाव नहीं बता सकता।"
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "अपनी दवा में किसी भी बदलाव के लिए कृपया किसी योग्य स्वास्थ्य विशेषज्ञ से परामर्श करें।"
    ),
    "I'm not sure how to interpret that.": "मुझे यह समझ नहीं आया।",
    "Could you please clarify?": "क्या आप कृपया स्पष्ट कर सकते हैं?",
    # Short clarifications the workflow offers when a reply is ambiguous
    # (kept in sync with _FIELD_CLARIFICATIONS in the conversational workflow).
    "Sorry, I didn't catch the product name.": (
        "क्षमा करें, मैं प्रोडक्ट का नाम समझ नहीं पाया।"
    ),
    "Which medicine were you using?": "आप कौन-सी दवा इस्तेमाल कर रहे थे?",
    "Sorry, could you describe the side effect in a few words?": (
        "क्षमा करें, क्या आप दुष्प्रभाव कुछ शब्दों में बता सकते हैं?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "बस पुष्टि के लिए - क्या आप अपना ही अनुभव बता रहे हैं, या किसी और का?"
    ),
    "Could you say your first and last name again?": (
        "क्या आप अपना पूरा नाम फिर से बता सकते हैं?"
    ),
    "How old are you, in years?": "आपकी उम्र कितनी है, वर्षों में?",
    "Could you tell me your age in years, or your date of birth?": (
        "क्या आप मुझे वर्षों में अपनी उम्र बता सकते हैं, या आपकी जन्मतिथि?"
    ),
    "Would it be alright if we contact you later - yes or no?": (
        "क्या हम बाद में आपसे संपर्क कर सकते हैं - हाँ या ना?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "आपके लिए क्या आसान है - फ़ोन, ईमेल, या दोनों?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "क्या आप फ़ोन नंबर अंकों में दोहरा सकते हैं?"
    ),
    "Could you repeat the email address?": (
        "क्या आप ईमेल पता फिर से बता सकते हैं?"
    ),
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "यह करीबन कब शुरू हुआ - जैसे आज, कल, या कुछ दिन पहले?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "क्या आप इसे गंभीर मानते हैं - हाँ, ना, या पक्का नहीं पता?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "क्या आपके लक्षण बेहतर हुए हैं, वैसे ही हैं, या और बिगड़ गए हैं?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "क्या इसके लिए आपने कोई चिकित्सा सहायता ली - हाँ या ना?"
    ),
    # Interview questions as the engine emits them (em-dash form). The Hindi
    # wording reuses the already-reviewed translations of the hyphen forms.
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "अब रोगी कैसा महसूस कर रहा है - क्या लक्षण बेहतर हुए हैं, वैसे ही हैं, या और बिगड़ गए हैं?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "अब आप कैसा महसूस कर रहे हैं - क्या आपके लक्षण बेहतर हुए हैं, वैसे ही हैं, या और बिगड़ गए हैं?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "आप फ़ोन, ईमेल, या दोनों से संपर्क करना चाहेंगे?"
    ),
    # Declined-contact-consent closing (each sentence, as split for translation).
    "Thank you for sharing this information.": (
        "यह जानकारी साझा करने के लिए धन्यवाद।"
    ),
    "We've recorded what you've provided.": (
        "हमने आपके द्वारा दी गई जानकारी दर्ज कर ली है।"
    ),
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "चूँकि आपने आगे संपर्क के लिए सहमति नहीं दी है, "
        "हम इन विवरणों का उपयोग करके आपसे संपर्क नहीं करेंगे।"
    ),
    # Confirmation read-back loop parked for human review (kept in sync with
    # CONFIRMATION_REVIEW_NOTICE in the conversational workflow). Honest
    # wording: some details could not be confirmed; nothing is claimed to be
    # fully saved, complete or submitted.
    "Thank you for your patience.": "आपके धैर्य के लिए धन्यवाद।",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "हम कुछ विवरणों की पुष्टि नहीं कर सके, इसलिए आपका मामला हमारी टीम के "
        "एक सदस्य द्वारा समीक्षा किया जाएगा।"
    ),
    # Product-scope notice for a non-Lantus suspected medicine (kept in sync
    # with _product_scope_message in the conversational workflow).
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "यह MedVaani प्रोटोटाइप केवल Lantus SoloStar के लिए "
        "कॉन्फ़िगर किया गया है।"
    ),
    "This prototype cannot continue this report.": (
        "यह प्रोटोटाइप इस रिपोर्ट को जारी नहीं रख सकता।"
    ),
    # ---- documented AE voice flow wording (kept in sync with
    # app.workflow.ae_engine._question_for and the step constants in
    # app.workflow.service.conversational) ----
    # Step 1 - fixed introduction (spoken once per session).
    "Thank you for taking the time to speak with us.": (
        "हमसे बात करने के लिए अपना समय देने के लिए धन्यवाद।"
    ),
    "I'm here to help record the event for review.": (
        "मैं समीक्षा के लिए घटना दर्ज करने में मदद के लिए यहाँ हूँ।"
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "मैं आपसे कुछ सवाल पूछूँगा, और हम समीक्षा के लिए एक रिपोर्ट तैयार "
        "करेंगे।"
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "मैं किसी स्वास्थ्य पेशेवर का विकल्प नहीं हूँ, और मैं चिकित्सकीय सलाह "
        "नहीं दे सकता।"
    ),
    # Step 2 - invite to describe the event.
    "Please tell me what happened, in your own words.": (
        "कृपया बताइए कि क्या हुआ, अपने शब्दों में।"
    ),
    "Take your time\u2014I'm listening.": (
        "आप अपने समय से बताइए\u2014मैं सुन रहा हूँ।"
    ),
    # Step 3 - medication question and its one-time empathy line.
    "Could you tell me the name of the medication involved?": (
        "क्या आप बता सकते हैं कि इसमें शामिल दवा का नाम क्या है?"
    ),
    "I'm sorry to hear that you've experienced this.": (
        "आपको यह अनुभव हुआ, यह जानकर मुझे खेद है।"
    ),
    # Step 4 - medication verification.
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "बस यह सुनिश्चित करने के लिए, क्या यह Lantus SoloStar है?"
    ),
    # Step 5 - reporter identification.
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "क्या मैं जान सकता हूँ कि क्या आप वही व्यक्ति हैं जिन्हें यह घटना हुई, "
        "या आप किसी और की ओर से यह बता रहे हैं?"
    ),
    # Step 6 - identity and contact questions.
    "Could you please tell me your full name?": (
        "क्या आप मुझे अपना पूरा नाम बता सकते हैं?"
    ),
    "Could you please share your contact number?": (
        "क्या आप अपना संपर्क नंबर बता सकते हैं?"
    ),
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "क्या आप मुझे रोगी का पहला और अंतिम नाम, साथ ही उनकी जन्मतिथि बता सकते हैं?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "क्या आप मुझे अपना पहला और अंतिम नाम, साथ ही आपकी जन्मतिथि बता सकते हैं?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "क्या आप मुझे रोगी की जन्मतिथि बता सकते हैं?"
    ),
    "Could you please tell me your date of birth?": (
        "क्या आप मुझे अपनी जन्मतिथि बता सकते हैं?"
    ),
    "Was it the name or the date of birth that was incorrect?": (
        "क्या नाम गलत था या जन्मतिथि?"
    ),
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "क्षमा करें, कृपया बताइए कि नाम या जन्मतिथि में से क्या गलत था।"
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "बस पुष्टि के लिए - क्या यह कोई और लक्षण है, या आपके द्वारा बताई गई बात "
        "में सुधार है?"
    ),
    "Sorry, please tell me a full date of birth.": (
        "क्षमा करें, कृपया पूरी जन्मतिथि बताइए।"
    ),
    # Empty-transcript retry (missing input, never an answer).
    "Sorry, I couldn't catch that. Could you please repeat it?": (
        "क्षमा करें, मुझे समझ नहीं आया। क्या आप कृपया दोबारा कह सकते हैं?"
    ),
    "I couldn't catch the email address. Could you please repeat it?": (
        "क्षमा करें, मुझे ईमेल पता समझ नहीं आया। क्या आप कृपया दोबारा बता सकते हैं?"
    ),
    "Could you please share the patient's contact number?": (
        "क्या आप रोगी का संपर्क नंबर बता सकते हैं?"
    ),
    "Could you please share the best contact number for you?": (
        "क्या आप अपने लिए सबसे अच्छा संपर्क नंबर बता सकते हैं?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "क्या आप संपर्क नंबर अंकों में दोहरा सकते हैं?"
    ),
    # Step 7 - consent.
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "अगर आगे कोई जानकारी चाहिए हो, तो क्या उचित टीम इस रिपोर्ट के संबंध "
        "में रोगी से संपर्क कर सकती है?"
    ),
    # Step 8 - conditional email question.
    "May I have the patient's email address, if available?": (
        "क्या रोगी का ईमेल पता मिल सकता है, यदि उपलब्ध हो तो?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "कृपया वह ईमेल पता बताइए जो आप हमें दर्ज करवाना चाहते हैं।"
    ),
    "Could you please spell out your email address, one letter or number at a time?": (
        "क्या आप अपना ईमेल पता एक-एक अक्षर या अंक में बता सकते हैं?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "आप '@' के लिए 'at the rate' और '.' के लिए 'dot' कह सकते हैं।"
    ),
    # Step 9 - severity screen.
    "Would you say that the event was severe or life-threatening?": (
        "क्या आप कहेंगे कि यह घटना गंभीर या जानलेवा थी?"
    ),
    # Step 10A - urgent safety response.
    "Please seek immediate medical attention now.": (
        "कृपया अभी तुरंत चिकित्सकीय सहायता लें।"
    ),
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "अगर आपने आगे संपर्क के लिए सहमति दी है, तो हम अतिरिक्त जानकारी बाद "
        "में एक सुविधाजनक समय पर एकत्र करवाएँगे।"
    ),
    # Step 10B.1 - incident date.
    "Could you tell me when the event occurred?": (
        "क्या आप बता सकते हैं कि यह घटना कब हुई?"
    ),
    # Step 10B.2 - additional information.
    "Is there anything else you'd like to share that may help us document the event?": (
        "क्या ऐसा कुछ और है जो आप साझा करना चाहेंगे, जो इस घटना को दर्ज "
        "करने में हमारी मदद कर सके?"
    ),
    "Of course.": "ज़रूर।",
    "Please tell me any additional details you'd like included.": (
        "कृपया वे सभी अतिरिक्त विवरण बताइए जो आप शामिल करवाना चाहते हैं।"
    ),
    # Step 12 - closing message (consent sentence only when consent stands).
    "Thank you for sharing this information with us.": (
        "हमारे साथ यह जानकारी साझा करने के लिए धन्यवाद।"
    ),
    "We've prepared the report for review.": (
        "हमने समीक्षा के लिए रिपोर्ट तैयार कर ली है।"
    ),
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "अगर आपने आगे संपर्क के लिए सहमति दी है, तो ज़रूरत पड़ने पर उचित टीम "
        "आपसे संपर्क कर सकती है।"
    ),
    # Step 6 read-backs (values are interpolated unchanged).
    "Is that right?": "क्या यह सही है?",
    "Have I got that right?": "क्या मैंने सही समझा?",
    "Have I recorded those correctly?": "क्या मैंने उन्हें सही ढंग से दर्ज किया है?",
}

# Parameterised templates. Values are interpolated exactly as captured (names,
# phone digits, labels) and never translated.
_HINDI_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (
        re.compile(r"^I heard (.+)\.$"),
        "मैंने {0} सुना।",
    ),
    # Step 6 read-backs: the captured value is interpolated unchanged.
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "मेरे पास आपका नाम {0} है।",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "मैंने आपके विवरण इस तरह दर्ज किए हैं: {0}।",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "मैंने इसे सही ढंग से दर्ज किया है या नहीं, यह सुनिश्चित करने के लिए "
        "वापस पढ़कर सुनाता हूँ: {0}।",
    ),
    (
        re.compile(r"^I have the number ending in (.+)\.$"),
        "नंबर का अंतिम हिस्सा {0} है।",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "क्षमा करें, मैं समझ नहीं पाया - कृपया सही {0} बताइए।",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "क्या आप {0} के बारे में थोड़ा और बता सकते हैं?",
    ),
    # DOB/age conflict clarification (kept in sync with
    # _DOB_AGE_CLARIFICATION in the conversational workflow).
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "आपकी जन्मतिथि ({0}) और आपकी आयु ({1} वर्ष) मेल नहीं खाते।",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "मैं कौन-सा दर्ज करूँ - आपकी जन्मतिथि या आपकी आयु?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "कृपया {0} के लिए उस दवा के निर्माता या अपने स्थानीय "
        "प्रतिकूल घटना रिपोर्टिंग चैनल से संपर्क करें।",
    ),
)

_CATALOGS: dict[str, tuple[dict[str, str], tuple[tuple[Pattern[str], str], ...]]] = {
    "hi-IN": (_HINDI_SENTENCES, _HINDI_PATTERNS),
}


def _split_sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_BOUNDARY.split(text.strip()) if sentence]


def _translate_sentence(
    sentence: str,
    catalog: tuple[dict[str, str], tuple[tuple[Pattern[str], str], ...]],
) -> Optional[str]:
    sentences, patterns = catalog
    exact = sentences.get(sentence)
    if exact is not None:
        return exact
    for pattern, template in patterns:
        match = pattern.match(sentence)
        if match:
            return template.format(*match.groups())
    return None


def translation_available(language: str | None) -> bool:
    """Whether a full translation catalog exists for ``language``."""
    return isinstance(language, str) and language in _CATALOGS


def translate(text: str, language: str | None) -> TranslationResult:
    """Return the spoken form of ``text`` in ``language``.

    Falls back to English (with ``translated=False``) whenever the language is
    English, unsupported here, or any sentence lacks a translation.
    """
    if not text or not text.strip():
        return TranslationResult(text, False, _ENGLISH)
    if not supports_language(language) or language == _ENGLISH:
        return TranslationResult(text, False, _ENGLISH)

    catalog = _CATALOGS.get(language or "")
    if catalog is None:
        return TranslationResult(text, False, _ENGLISH)

    translated_sentences: list[str] = []
    for sentence in _split_sentences(text):
        translated = _translate_sentence(sentence, catalog)
        if translated is None:
            return TranslationResult(text, False, _ENGLISH)
        translated_sentences.append(translated)
    if not translated_sentences:
        return TranslationResult(text, False, _ENGLISH)
    return TranslationResult(" ".join(translated_sentences), True, language)


__all__ = ["TranslationResult", "translate", "translation_available"]
