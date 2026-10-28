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
- All ten Prisma/Timbre languages are populated today; English is a
  passthrough. The catalog is intentionally a plain dict so languages can be
  added without touching callers.
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
    "Could you say the date of birth again?": (
        "क्या आप जन्मतिथि दोबारा बता सकते हैं?"
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
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "यह पूरा पता है, तो हाँ कहें - अगर इसमें और अक्षर बाकी हैं, तो कृपया पता दोबारा बताएं।"
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "यह सही नहीं है, तो कृपया इसे फिर से एक-एक अक्षर या अंक बताकर बनाएं।"
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "हमें नाम की पुष्टि नहीं हो सकी।"
    ),
    "We could not confirm the phone number.": (
        "हमें फ़ोन नंबर की पुष्टि नहीं हो सकी।"
    ),
    "We could not confirm the contact number.": (
        "हमें संपर्क नंबर की पुष्टि नहीं हो सकी।"
    ),
    "You can say it again now, or our team will collect it during review.": (
        "आप इसे अभी दोबारा बता सकते हैं, या समीक्षा के दौरान हमारी टीम इसे ले लेगी।"
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
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "रिपोर्ट सही ढंग से दर्ज करने में मदद के लिए, क्या आप बता सकते हैं कि "
        "इसमें शामिल दवा का नाम क्या है?"
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
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "मुझे महीना ठीक से सुनाई नहीं दिया - कृपया जन्मतिथि दिन, महीने और साल के रूप में बताएं।"
    ),
    # Empty-transcript retry (missing input, never an answer).
    "Sorry, I couldn't catch that.": (
        "क्षमा करें, मुझे समझ नहीं आया।"
    ),
    "Could you please repeat it?": (
        "क्या आप कृपया दोबारा कह सकते हैं?"
    ),
    "I couldn't catch the email address.": (
        "क्षमा करें, मुझे ईमेल पता समझ नहीं आया।"
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
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
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
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "पता {0} के रूप में पढ़ा गया है।",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "पता अभी भी {0} ही है।",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "मैंने आपका ईमेल पता {0} समझा।",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "मैंने आपका ईमेल पता {0} सुना।",
    ),
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
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "मुझे {0} अंकों का नंबर मिला है, जिसका अंतिम हिस्सा {1} है।",
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

# ---- Bengali (bn-IN) ----------------------------------------------------------

_BENGALI_SENTENCES: dict[str, str] = {
    "Thanks for that.": "এর জন্য ধন্যবাদ।",
    "Got it, thank you.": "বুঝেছি, ধন্যবাদ।",
    "Okay, thank you.": "ঠিক আছে, ধন্যবাদ।",
    "Thanks for the details.": "বিবরণ দেওয়ার জন্য ধন্যবাদ।",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "আমাকে বলার জন্য ধন্যবাদ, এবং আপনি এসব ভোগ করছেন জেনে আমি দুঃখিত।"
    ),
    "I'll take a few details so we can record it safely.": (
        "আমি কিছু তথ্য নেব যাতে আমরা এটি নিরাপদে রেকর্ড করতে পারি।"
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "গুরুত্বপূর্ণ: জানানো পার্শ্বপ্রতিক্রিয়াগুলির মূল্যায়ন ও চিকিৎসার জন্য "
        "অনুগ্রহ করে একজন যোগ্য চিকিৎসক বা স্বাস্থ্যসেবা পেশাদারের পরামর্শ নিন।"
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "চিকিৎসকের পরামর্শ ছাড়া নিজে নিজে রোগ নির্ণয় করবেন না বা আপনার ওষুধ পরিবর্তন করবেন না।"
    ),
    "Which medicine or product do you think may have caused this?": (
        "আপনার মনে হয় কোন ওষুধ বা পণ্য এটি ঘটাতে পারে?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "আপনি কী পার্শ্বপ্রতিক্রিয়া বা প্রতিক্রিয়া লক্ষ্য করেছেন বলতে পারবেন?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "আপনি কি সেই ব্যক্তি যিনি এটি অনুভব করেছেন, নাকি অন্য কারো হয়ে জানাচ্ছেন?"
    ),
    "What is the patient's full name?": "রোগীর পুরো নাম কী?",
    "May I have your first and last name?": "আপনার প্রথম ও শেষ নাম জানতে পারি?",
    "How old is the patient, or what is their date of birth?": (
        "রোগীর বয়স কত, বা তাঁর জন্ম তারিখ কী?"
    ),
    "How old are you, or what is your date of birth?": (
        "আপনার বয়স কত, বা আপনার জন্ম তারিখ কী?"
    ),
    "And what is your name?": "আর আপনার নাম কী?",
    "If we need a little more information later, is it alright if we contact you?": (
        "পরে যদি আমাদের আর একটু তথ্য দরকার হয়, তাহলে কি আমরা আপনার সঙ্গে যোগাযোগ করতে পারি?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "আপনি ফোনে যোগাযোগ পছন্দ করবেন নাকি ইমেলে?"
    ),
    "What phone number would you like us to use?": "আমরা কোন ফোন নম্বর ব্যবহার করব?",
    "What email address would you like us to use?": "আমরা কোন ইমেল ঠিকানা ব্যবহার করব?",
    "When did the side effect first start?": "পার্শ্বপ্রতিক্রিয়াটি প্রথম কখন শুরু হয়েছিল?",
    "Would you consider this problem serious?": "আপনি কি এই সমস্যাটিকে গুরুতর মনে করেন?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "রোগী এখন কেমন অনুভব করছেন - উপসর্গগুলি কি ভালো হয়েছে, আগের মতোই আছে, নাকি আরও খারাপ হয়েছে?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "আপনি এখন কেমন অনুভব করছেন - আপনার উপসর্গগুলি কি ভালো হয়েছে, আগের মতোই আছে, নাকি আরও খারাপ হয়েছে?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "দুঃখিত, আমি বুঝতে পারিনি - এটি সঠিক হলে হ্যাঁ বলুন, আর পরিবর্তন করতে চাইলে না বলুন।"
    ),
    "Is that correct?": "এটি কি সঠিক?",
    "Thank you - I've recorded everything we need for now.": (
        "ধন্যবাদ - এই মুহূর্তে যা প্রয়োজন তা আমি রেকর্ড করে নিয়েছি।"
    ),
    "A member of our team will follow up if needed.": (
        "প্রয়োজন হলে আমাদের দলের একজন সদস্য আপনার সঙ্গে যোগাযোগ করবেন।"
    ),
    "Take care.": "নিজের যত্ন নিন।",
    "Thank you.": "ধন্যবাদ।",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "শেষ করার আগে, কী ঘটেছিল সে সম্পর্কে আর কিছু বলতে চান?"
    ),
    "Thank you for sharing that.": "সেটি জানানোর জন্য ধন্যবাদ।",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "এটি যদি চিকিৎসা জরুরি অবস্থা হয়, তাহলে অনুগ্রহ করে তাৎক্ষণিক চিকিৎসা সহায়তা নিন "
        "বা আপনার স্থানীয় জরুরি সেবায় কল করুন।"
    ),
    "This appears potentially urgent.": "এটি সম্ভাব্য জরুরি মনে হচ্ছে।",
    "A healthcare professional should be contacted immediately.": (
        "অবিলম্বে একজন স্বাস্থ্যসেবা পেশাদারের সঙ্গে যোগাযোগ করা উচিত।"
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "আমি ব্যক্তিগত ডোজ, চিকিৎসার পরামর্শ বা ওষুধ পরিবর্তন জানাতে পারি না।"
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "আপনার ওষুধের রুটিনে যেকোনো পরিবর্তনের জন্য অনুগ্রহ করে একজন যোগ্য "
        "স্বাস্থ্যসেবা পেশাদারের পরামর্শ নিন।"
    ),
    "I'm not sure how to interpret that.": "আমি এর অর্থ বুঝতে পারিনি।",
    "Could you please clarify?": "আপনি কি অনুগ্রহ করে পরিষ্কার করে বলতে পারেন?",
    "Sorry, I didn't catch the product name.": "দুঃখিত, আমি পণ্যের নাম বুঝতে পারিনি।",
    "Which medicine were you using?": "আপনি কোন ওষুধ ব্যবহার করছিলেন?",
    "Sorry, could you describe the side effect in a few words?": (
        "দুঃখিত, আপনি কি পার্শ্বপ্রতিক্রিয়াটি কয়েকটি শব্দে বর্ণনা করতে পারেন?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "নিশ্চিত হওয়ার জন্য - আপনি কি নিজের অভিজ্ঞতা জানাচ্ছেন, নাকি অন্য কারো?"
    ),
    "Could you say your first and last name again?": (
        "আপনি কি আবার আপনার প্রথম ও শেষ নাম বলতে পারেন?"
    ),
    "How old are you, in years?": "আপনার বয়স কত, বছরে?",
    "Could you tell me your age in years, or your date of birth?": (
        "আপনি কি বছরে আপনার বয়স বা আপনার জন্ম তারিখ বলতে পারেন?"
    ),
    "Could you say the date of birth again?": "আপনি কি জন্ম তারিখ আবার বলতে পারেন?",
    "Would it be alright if we contact you later - yes or no?": (
        "আমরা কি পরে আপনার সঙ্গে যোগাযোগ করতে পারি - হ্যাঁ নাকি না?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "আপনার জন্য কোনটি সহজ - ফোন, ইমেল, নাকি দুটোই?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "আপনি কি ফোন নম্বরটি অঙ্ক ধরে ধরে বলতে পারেন?"
    ),
    "Could you repeat the email address?": "আপনি কি ইমেল ঠিকানা আবার বলতে পারেন?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "এটি মোটামুটি কখন শুরু হয়েছিল - যেমন আজ, গতকাল, নাকি কয়েক দিন আগে?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "আপনি কি এটিকে গুরুতর মনে করেন - হ্যাঁ, না, নাকি নিশ্চিত নয়?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "আপনার উপসর্গগুলি কি এখন ভালো, আগের মতোই, নাকি খারাপ?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "এর জন্য কি আপনি কোনো চিকিৎসা পেয়েছেন - হ্যাঁ নাকি না?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "রোগী এখন কেমন অনুভব করছেন - উপসর্গগুলি কি ভালো হয়েছে, আগের মতোই আছে, নাকি আরও খারাপ হয়েছে?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "আপনি এখন কেমন অনুভব করছেন - আপনার উপসর্গগুলি কি ভালো হয়েছে, আগের মতোই আছে, নাকি আরও খারাপ হয়েছে?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "আপনি ফোনে, ইমেলে, নাকি দুটোতেই যোগাযোগ পছন্দ করবেন?"
    ),
    "Thank you for sharing this information.": "এই তথ্য ভাগ করে নেওয়ার জন্য ধন্যবাদ।",
    "We've recorded what you've provided.": "আপনি যা দিয়েছেন তা আমরা রেকর্ড করেছি।",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "যেহেতু আপনি পরবর্তী যোগাযোগের জন্য সম্মতি দেননি, আমরা এই বিবরণ ব্যবহার "
        "করে আপনার সঙ্গে যোগাযোগ করব না।"
    ),
    "Thank you for your patience.": "আপনার ধৈর্যের জন্য ধন্যবাদ।",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "আমরা কিছু বিবরণ নিশ্চিত করতে পারিনি, তাই আপনার কেসটি আমাদের দলের "
        "একজন সদস্য পর্যালোচনা করবেন।"
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "এটি সম্পূর্ণ ঠিকানা হলে হ্যাঁ বলুন - আরও অক্ষর বাকি থাকলে অনুগ্রহ করে ঠিকানাটি আবার বানান।"
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "এটি সঠিক না হলে, অনুগ্রহ করে এটি আবার এক-একটি করে অক্ষর বা সংখ্যা বলে বানান।"
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "আমরা নামটি নিশ্চিত করতে পারিনি।"
    ),
    "We could not confirm the phone number.": (
        "আমরা ফোন নম্বরটি নিশ্চিত করতে পারিনি।"
    ),
    "We could not confirm the contact number.": (
        "আমরা যোগাযোগের নম্বরটি নিশ্চিত করতে পারিনি।"
    ),
    "You can say it again now, or our team will collect it during review.": (
        "আপনি এখনই এটি আবার বলতে পারেন, অথবা পর্যালোচনার সময় আমাদের দল এটি সংগ্রহ করবে।"
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "এই MedVaani প্রোটোটাইপ শুধুমাত্র Lantus SoloStar-এর জন্য কনফিগার করা আছে।"
    ),
    "This prototype cannot continue this report.": "এই প্রোটোটাইপ এই রিপোর্টটি চালিয়ে যেতে পারবে না।",
    "Thank you for taking the time to speak with us.": "আমাদের সঙ্গে কথা বলার সময় নেওয়ার জন্য ধন্যবাদ।",
    "I'm here to help record the event for review.": (
        "পর্যালোচনার জন্য ঘটনাটি রেকর্ড করতে সাহায্য করার জন্য আমি এখানে আছি।"
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "আমি আপনাকে কয়েকটি প্রশ্ন করব, এবং আমরা পর্যালোচনার জন্য একটি রিপোর্ট তৈরি করব।"
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "আমি কোনো স্বাস্থ্যসেবা পেশাদারের বিকল্প নই, এবং আমি চিকিৎসা পরামর্শ দিতে পারি না।"
    ),
    "Please tell me what happened, in your own words.": "অনুগ্রহ করে নিজের ভাষায় বলুন কী ঘটেছিল।",
    "Take your time\u2014I'm listening.": "আপনার সময় নিন\u2014আমি শুনছি।",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "আপনার রিপোর্টটি সঠিকভাবে রেকর্ড করতে সাহায্য করার জন্য, আপনি কি বলতে "
        "পারেন সংশ্লিষ্ট ওষুধের নাম কী?"
    ),
    "I'm sorry to hear that you've experienced this.": "আপনি এটি অনুভব করেছেন জেনে আমি দুঃখিত।",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "শুধু নিশ্চিত হওয়ার জন্য, এটি কি Lantus SoloStar?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "আমি কি জানতে পারি, আপনি কি সেই ব্যক্তি যিনি এই ঘটনা অনুভব করেছেন, "
        "নাকি অন্য কারো হয়ে জানাচ্ছেন?"
    ),
    "Could you please tell me your full name?": "আপনি কি অনুগ্রহ করে আপনার পুরো নাম বলতে পারেন?",
    "Could you please share your contact number?": "আপনি কি অনুগ্রহ করে আপনার যোগাযোগ নম্বর বলতে পারেন?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "আপনি কি অনুগ্রহ করে রোগীর প্রথম ও শেষ নাম, এবং তাঁর জন্ম তারিখ বলতে পারেন?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "আপনি কি অনুগ্রহ করে আপনার প্রথম ও শেষ নাম, এবং আপনার জন্ম তারিখ বলতে পারেন?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "আপনি কি অনুগ্রহ করে রোগীর জন্ম তারিখ বলতে পারেন?"
    ),
    "Could you please tell me your date of birth?": "আপনি কি অনুগ্রহ করে আপনার জন্ম তারিখ বলতে পারেন?",
    "Was it the name or the date of birth that was incorrect?": "কি নামটি ভুল ছিল নাকি জন্ম তারিখ?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "দুঃখিত, অনুগ্রহ করে বলুন নাম নাকি জন্ম তারিখ - কোনটি ভুল ছিল।"
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "নিশ্চিত হওয়ার জন্য - এটি কি অন্য কোনো উপসর্গ, নাকি আপনার বর্ণনার সংশোধন?"
    ),
    "Sorry, please tell me a full date of birth.": "দুঃখিত, অনুগ্রহ করে সম্পূর্ণ জন্ম তারিখ বলুন।",
    "Sorry, I couldn't catch that.": (
        "দুঃখিত, আমি বুঝতে পারিনি।"
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "মাসটি স্পষ্টভাবে শুনতে পাইনি - অনুগ্রহ করে জন্মতারিখ দিন, মাস এবং বছর হিসেবে বলুন।"
    ),
    "Could you please repeat it?": (
        "আপনি কি অনুগ্রহ করে আবার বলতে পারেন?"
    ),
    "I couldn't catch the email address.": (
        "আমি ইমেল ঠিকানা বুঝতে পারিনি।"
    ),
    "Could you please share the patient's contact number?": (
        "আপনি কি অনুগ্রহ করে রোগীর যোগাযোগ নম্বর বলতে পারেন?"
    ),
    "Could you please share the best contact number for you?": (
        "আপনি কি আপনার জন্য সবচেয়ে ভালো যোগাযোগ নম্বর বলতে পারেন?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "আপনি কি যোগাযোগ নম্বরটি অঙ্ক ধরে ধরে বলতে পারেন?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "পরবর্তী কোনো তথ্য প্রয়োজন হলে, উপযুক্ত দল কি এই রিপোর্ট সম্পর্কে রোগীর "
        "সঙ্গে যোগাযোগ করতে পারবে?"
    ),
    "May I have the patient's email address, if available?": "রোগীর ইমেল ঠিকানা কি পাওয়া সম্ভব, থাকলে?",
    "Please tell me the email address you'd like us to record.": (
        "অনুগ্রহ করে সেই ইমেল ঠিকানা বলুন যা আপনি রেকর্ড করাতে চান।"
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "আপনি কি আপনার ইমেল ঠিকানা একে একে একটি করে অক্ষর বা অঙ্ক বলতে পারেন?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "আপনি @-এর জন্য 'at the rate' এবং বিন্দুর জন্য 'dot' বলতে পারেন।"
    ),
    "Would you say that the event was severe or life-threatening?": (
        "আপনি কি বলবেন যে ঘটনাটি ছিল গুরুতর বা জীবনঘাতী?"
    ),
    "Please seek immediate medical attention now.": "অনুগ্রহ করে এখনই তাৎক্ষণিক চিকিৎসা সহায়তা নিন।",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "আপনি যদি ইতিমধ্যে পরবর্তী যোগাযোগের জন্য সম্মতি দিয়ে থাকেন, তাহলে আমরা "
        "আরও তথ্য সংগ্রহের ব্যবস্থা করব একটি সুবিধাজনক সময়ে।"
    ),
    "Could you tell me when the event occurred?": "আপনি কি বলতে পারেন ঘটনাটি কখন ঘটেছিল?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "ঘটনাটি নথিভুক্ত করতে সাহায্য করতে পারে এমন আর কিছু আছে কি যা আপনি ভাগ করতে চান?"
    ),
    "Of course.": "অবশ্যই।",
    "Please tell me any additional details you'd like included.": (
        "অনুগ্রহ করে সেই সব অতিরিক্ত বিবরণ বলুন যা আপনি অন্তর্ভুক্ত করাতে চান।"
    ),
    "Thank you for sharing this information with us.": "আমাদের সঙ্গে এই তথ্য ভাগ করে নেওয়ার জন্য ধন্যবাদ।",
    "We've prepared the report for review.": "আমরা পর্যালোচনার জন্য রিপোর্ট তৈরি করেছি।",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "আপনি যদি পরবর্তী যোগাযোগের জন্য সম্মতি দিয়ে থাকেন, তাহলে প্রয়োজনে "
        "উপযুক্ত দল আপনার সঙ্গে যোগাযোগ করতে পারে।"
    ),
    "Is that right?": "এটি কি ঠিক আছে?",
    "Have I got that right?": "আমি কি সঠিক বুঝেছি?",
    "Have I recorded those correctly?": "আমি কি সেগুলি সঠিকভাবে রেকর্ড করেছি?",
}

_BENGALI_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "ঠিকানাটি {0} হিসেবে পড়া হয়েছে।",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "ঠিকানাটি এখনও {0} হিসেবেই রয়েছে।",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "আমি আপনার ইমেল ঠিকানা {0} বুঝেছি।",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "আমি আপনার ইমেল ঠিকানা {0} শুনেছি।",
    ),
    (re.compile(r"^I heard (.+)\.$"), "আমি {0} শুনেছি।"),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "আপনার নাম {0} বলে আমার কাছে রেকর্ড করা আছে।",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "আপনার বিবরণ আমি এভাবে রেকর্ড করেছি: {0}।",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "সঠিকভাবে রেকর্ড করেছি কি না, তা নিশ্চিত হওয়ার জন্য আমি ফিরে পড়ে শোনাচ্ছি: {0}।",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "আমি {0} সংখ্যার নম্বর পেয়েছি, যার শেষাংশ {1}।",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "দুঃখিত, আমি বুঝতে পারিনি - অনুগ্রহ করে সঠিক {0} বলুন।",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "আপনি কি {0} সম্পর্কে আর একটু বলতে পারেন?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "আপনার জন্ম তারিখ ({0}) এবং আপনার বয়স ({1} বছর) মিলছে না।",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "আমি কোনটি রেকর্ড করব - আপনার জন্ম তারিখ নাকি আপনার বয়স?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0}-এর জন্য অনুগ্রহ করে ওষুধের নির্মাতা বা আপনার স্থানীয় "
        "প্রতিকূল ঘটনা রিপোর্টিং চ্যানেলের সঙ্গে যোগাযোগ করুন।",
    ),
)

# ---- Gujarati (gu-IN) ---------------------------------------------------------

_GUJARATI_SENTENCES: dict[str, str] = {
    "Thanks for that.": "આ માટે આભાર.",
    "Got it, thank you.": "સમજાઈ ગયું, આભાર.",
    "Okay, thank you.": "ઠીક છે, આભાર.",
    "Thanks for the details.": "વિગતો આપવા બદલ આભાર.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "મને જણાવવા બદલ આભાર, અને તમે આ બધું અનુભવો છો તે માટે મને દુઃખ છે."
    ),
    "I'll take a few details so we can record it safely.": (
        "હું થોડી વિગતો લઈશ જેથી આપણે તેને સુરક્ષિત રીતે નોંધી શકીએ."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "મહત્વપૂર્ણ: જણાવેલા આડઅસરોનું મૂલ્યાંકન અને સારવાર માટે કૃપા કરીને "
        "લાયક ડૉક્ટર અથવા આરોગ્યસેવા વ્યાવસાયિકની સલાહ લો."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "તબીબી સલાહ વગર જાતે નિદાન કરશો નહીં કે તમારી દવા બદલશો નહીં."
    ),
    "Which medicine or product do you think may have caused this?": (
        "તમારા મતે કઈ દવા કે પ્રોડક્ટથી આ થયું હોઈ શકે?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "તમે કઈ આડઅસર કે પ્રતિક્રિયા જોઈ તે મને જણાવી શકશો?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "શું તમે એ વ્યક્તિ છો જેને આ થયું, કે તમે બીજા કોઈ વતી જણાવી રહ્યા છો?"
    ),
    "What is the patient's full name?": "દર્દીનું પૂરું નામ શું છે?",
    "May I have your first and last name?": "શું હું તમારું પ્રથમ અને છેલ્લું નામ જાણી શકું?",
    "How old is the patient, or what is their date of birth?": (
        "દર્દીની ઉંમર કેટલી છે, અથવા તેમની જન્મ તારીખ શું છે?"
    ),
    "How old are you, or what is your date of birth?": (
        "તમારી ઉંમર કેટલી છે, અથવા તમારી જન્મ તારીખ શું છે?"
    ),
    "And what is your name?": "અને તમારું નામ શું છે?",
    "If we need a little more information later, is it alright if we contact you?": (
        "પછીથી જો અમને થોડી વધુ માહિતી જોઈએ, તો શું અમે તમારો સંપર્ક કરી શકીએ?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "તમે ફોન દ્વારા સંપર્ક પસંદ કરશો કે ઈમેલ દ્વારા?"
    ),
    "What phone number would you like us to use?": "અમે કયો ફોન નંબર વાપરીએ?",
    "What email address would you like us to use?": "અમે કયું ઈમેલ સરનામું વાપરીએ?",
    "When did the side effect first start?": "આડઅસર પહેલીવાર ક્યારે શરૂ થઈ?",
    "Would you consider this problem serious?": "શું તમે આ સમસ્યાને ગંભીર ગણશો?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "દર્દીને હવે કેવું લાગે છે - લક્ષણો સુધર્યા છે, એવા જ છે, કે વધુ બગડ્યા છે?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "તમને હવે કેવું લાગે છે - તમારા લક્ષણો સુધર્યા છે, એવા જ છે, કે વધુ બગડ્યા છે?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "માફ કરશો, મને સમજાયું નહીં - સાચું હોય તો હા કહો, અને બદલવું હોય તો ના કહો."
    ),
    "Is that correct?": "શું તે સાચું છે?",
    "Thank you - I've recorded everything we need for now.": (
        "આભાર - હમણાં માટે જરૂરી બધું મેં નોંધી લીધું છે."
    ),
    "A member of our team will follow up if needed.": (
        "જરૂર પડે તો અમારી ટીમના સભ્ય તમારો સંપર્ક કરશે."
    ),
    "Take care.": "તમારું ધ્યાન રાખજો.",
    "Thank you.": "આભાર.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "પૂરું કરતા પહેલાં, શું થયું તે વિશે બીજું કંઈ કહેવું છે?"
    ),
    "Thank you for sharing that.": "તે જણાવવા બદલ આભાર.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "આ તબીબી ઇમરજન્સી હોય તો, કૃપા કરીને તાત્કાલિક તબીબી સહાય મેળવો "
        "અથવા તમારી સ્થાનિક ઇમરજન્સી સેવાને કૉલ કરો."
    ),
    "This appears potentially urgent.": "આ સંભવિત રીતે તાકીદનું લાગે છે.",
    "A healthcare professional should be contacted immediately.": (
        "તરત જ કોઈ આરોગ્યસેવા વ્યાવસાયિકનો સંપર્ક કરવો જોઈએ."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "હું વ્યક્તિગત ડોઝ, સારવારની ભલામણો કે દવામાં ફેરફાર ન જણાવી શકું."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "તમારી દવામાં કોઈપણ ફેરફાર માટે કૃપા કરીને લાયક આરોગ્યસેવા "
        "વ્યાવસાયિકની સલાહ લો."
    ),
    "I'm not sure how to interpret that.": "મને તેનો અર્થ સમજાયો નહીં.",
    "Could you please clarify?": "શું તમે કૃપા કરીને સ્પષ્ટ કરી શકશો?",
    "Sorry, I didn't catch the product name.": "માફ કરશો, મને પ્રોડક્ટનું નામ સમજાયું નહીં.",
    "Which medicine were you using?": "તમે કઈ દવા વાપરતા હતા?",
    "Sorry, could you describe the side effect in a few words?": (
        "માફ કરશો, શું તમે આડઅસર થોડા શબ્દોમાં વર્ણવી શકશો?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "ખાતરી માટે - શું તમે પોતાનો અનુભવ જણાવો છો, કે બીજા કોઈનો?"
    ),
    "Could you say your first and last name again?": (
        "શું તમે તમારું પ્રથમ અને છેલ્લું નામ ફરી કહી શકશો?"
    ),
    "How old are you, in years?": "તમારી ઉંમર કેટલી છે, વર્ષમાં?",
    "Could you tell me your age in years, or your date of birth?": (
        "શું તમે વર્ષમાં તમારી ઉંમર કે તમારી જન્મ તારીખ જણાવી શકશો?"
    ),
    "Could you say the date of birth again?": "શું તમે જન્મ તારીખ ફરી કહી શકશો?",
    "Would it be alright if we contact you later - yes or no?": (
        "શું અમે પછીથી તમારો સંપર્ક કરી શકીએ - હા કે ના?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "તમારા માટે શું સરળ છે - ફોન, ઈમેલ, કે બંને?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "શું તમે ફોન નંબર અંક પ્રમાણે દોહરાવી શકશો?"
    ),
    "Could you repeat the email address?": "શું તમે ઈમેલ સરનામું ફરી કહી શકશો?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "તે લગભગ ક્યારે શરૂ થયું - જેમ કે આજે, ગઈકાલે, કે થોડા દિવસ પહેલાં?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "શું તમે તેને ગંભીર ગણો છો - હા, ના, કે ખાતરી નથી?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "તમારા લક્ષણો હવે સારા છે, એવા જ છે, કે વધુ ખરાબ?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "શું તમે તેના માટે કોઈ તબીબી સારવાર લીધી - હા કે ના?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "દર્દીને હવે કેવું લાગે છે - લક્ષણો સુધર્યા છે, એવા જ છે, કે વધુ બગડ્યા છે?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "તમને હવે કેવું લાગે છે - તમારા લક્ષણો સુધર્યા છે, એવા જ છે, કે વધુ બગડ્યા છે?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "તમે ફોન દ્વારા, ઈમેલ દ્વારા, કે બંને દ્વારા સંપર્ક પસંદ કરશો?"
    ),
    "Thank you for sharing this information.": "આ માહિતી શેર કરવા બદલ આભાર.",
    "We've recorded what you've provided.": "અમે તમે આપેલી માહિતી નોંધી લીધી છે.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "તમે પછીના સંપર્ક માટે સંમતિ આપી નથી, તેથી અમે આ વિગતો વાપરીને "
        "તમારો સંપર્ક નહીં કરીએ."
    ),
    "Thank you for your patience.": "તમારા ધીરજ બદલ આભાર.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "અમે કેટલીક વિગતોની પુષ્ટિ ન કરી શક્યા, તેથી તમારો કેસ અમારી ટીમના "
        "સભ્ય દ્વારા સમીક્ષા કરવામાં આવશે."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "તે સંપૂર્ણ સરનામું હોય તો હા કહો - જો વધુ અક્ષરો બાકી હોય તો કૃપા કરીને સરનામું ફરી લખો."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "જો તે સાચું નથી, તો કૃપા કરીને તેને ફરી એક-એક અક્ષર કે આંકડા સાથે લખો."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "અમે નામની પુષ્ટિ કરી શક્યા નથી."
    ),
    "We could not confirm the phone number.": (
        "અમે ફોન નંબરની પુષ્ટિ કરી શક્યા નથી."
    ),
    "We could not confirm the contact number.": (
        "અમે સંપર્ક નંબરની પુષ્ટિ કરી શક્યા નથી."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "તમે તેને હમણાં ફરી કહી શકો છો, અથવા સમીક્ષા દરમિયાન અમારી ટીમ તે લઈ લેશે."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "આ MedVaani પ્રોટોટાઇપ ફક્ત Lantus SoloStar માટે ગોઠવવામાં આવ્યું છે."
    ),
    "This prototype cannot continue this report.": "આ પ્રોટોટાઇપ આ રિપોર્ટ ચાલુ રાખી શકતું નથી.",
    "Thank you for taking the time to speak with us.": "અમારી સાથે વાત કરવા સમય આપવા બદલ આભાર.",
    "I'm here to help record the event for review.": (
        "સમીક્ષા માટે ઘટના નોંધવામાં મદદ કરવા હું અહીં છું."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "હું તમને થોડા પ્રશ્નો પૂછીશ, અને અમે સમીક્ષા માટે એક રિપોર્ટ તૈયાર કરીશું."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "હું કોઈ આરોગ્યસેવા વ્યાવસાયિકનો વિકલ્પ નથી, અને હું તબીબી સલાહ આપી શકતો નથી."
    ),
    "Please tell me what happened, in your own words.": "કૃપા કરીને પોતાના શબ્દોમાં કહો કે શું થયું.",
    "Take your time\u2014I'm listening.": "તમારો સમય લો\u2014હું સાંભળી રહ્યો છું.",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "તમારો રિપોર્ટ સાચી રીતે નોંધવામાં મદદ માટે, શું તમે જણાવી શકશો "
        "કે સામેલ દવાનું નામ શું છે?"
    ),
    "I'm sorry to hear that you've experienced this.": "તમે આ અનુભવ્યું તે જાણીને મને દુઃખ છે.",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "ખાતરી કરવા કે મારી પાસે સાચી દવા છે, શું તે Lantus SoloStar છે?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "શું હું જાણી શકું કે શું તમે એ વ્યક્તિ છો જેને આ ઘટના થઈ, કે તમે "
        "બીજા કોઈ વતી જણાવો છો?"
    ),
    "Could you please tell me your full name?": "શું તમે મને તમારું પૂરું નામ જણાવી શકશો?",
    "Could you please share your contact number?": "શું તમે તમારો સંપર્ક નંબર જણાવી શકશો?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "શું તમે મને દર્દીનું પ્રથમ અને છેલ્લું નામ, સાથે તેમની જન્મ તારીખ જણાવી શકશો?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "શું તમે મને તમારું પ્રથમ અને છેલ્લું નામ, સાથે તમારી જન્મ તારીખ જણાવી શકશો?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "શું તમે મને દર્દીની જન્મ તારીખ જણાવી શકશો?"
    ),
    "Could you please tell me your date of birth?": "શું તમે મને તમારી જન્મ તારીખ જણાવી શકશો?",
    "Was it the name or the date of birth that was incorrect?": "નામ ખોટું હતું કે જન્મ તારીખ?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "માફ કરશો, કૃપા કરીને જણાવો કે નામ કે જન્મ તારીખમાંથી શું ખોટું હતું."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "ખાતરી માટે - શું તે બીજું કોઈ લક્ષણ હતું, કે તમે વર્ણવેલી બાબતમાં સુધારો?"
    ),
    "Sorry, please tell me a full date of birth.": "માફ કરશો, કૃપા કરીને પૂરી જન્મ તારીખ કહો.",
    "Sorry, I couldn't catch that.": (
        "માફ કરશો, મને સમજાયું નહીં."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "મને મહિનો સ્પષ્ટપણે સંભળાયો નથી - કૃપા કરીને જન્મતારીખ દિવસ, મહિનો અને વર્ષ તરીકે કહો."
    ),
    "Could you please repeat it?": (
        "શું તમે કૃપા કરીને ફરી કહી શકશો?"
    ),
    "I couldn't catch the email address.": (
        "મને ઈમેલ સરનામું સમજાયું નહીં."
    ),
    "Could you please share the patient's contact number?": (
        "શું તમે દર્દીનો સંપર્ક નંબર જણાવી શકશો?"
    ),
    "Could you please share the best contact number for you?": (
        "શું તમે તમારા માટે શ્રેષ્ઠ સંપર્ક નંબર જણાવી શકશો?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "શું તમે સંપર્ક નંબર અંક પ્રમાણે દોહરાવી શકશો?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "પછીથી કોઈ માહિતી જોઈએ તો, યોગ્ય ટીમ શું આ રિપોર્ટ અંગે દર્દીનો "
        "સંપર્ક કરી શકશે?"
    ),
    "May I have the patient's email address, if available?": (
        "દર્દીનું ઈમેલ સરનામું મળી શકે, ઉપલબ્ધ હોય તો?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "કૃપા કરીને તે ઈમેલ સરનામું કહો જે તમે નોંધાવવા માંગો છો."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "શું તમે તમારું ઈમેલ સરનામું એક-એક અક્ષર કે અંકમાં કહી શકશો?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "તમે @ માટે 'at the rate' અને ટપકાં માટે 'dot' કહી શકો છો."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "શું તમે કહેશો કે ઘટના ગંભીર કે જીવલેણ હતી?"
    ),
    "Please seek immediate medical attention now.": "કૃપા કરીને હમણાં જ તાત્કાલિક તબીબી સહાય મેળવો.",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "તમે પછીના સંપર્ક માટે સંમતિ આપી હોય, તો અમે વધારાની માહિતી વધુ "
        "અનુકૂળ સમયે એકત્રિત કરાવીશું."
    ),
    "Could you tell me when the event occurred?": "શું તમે જણાવી શકશો કે ઘટના ક્યારે બની?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "ઘટના નોંધવામાં મદદ કરી શકે તેવું બીજું કંઈ છે કે તમે શેર કરવા માંગો?"
    ),
    "Of course.": "ચોક્કસ.",
    "Please tell me any additional details you'd like included.": (
        "કૃપા કરીને તે બધી વધારાની વિગતો કહો જે તમે સામેલ કરાવવા માંગો છો."
    ),
    "Thank you for sharing this information with us.": "અમારી સાથે આ માહિતી શેર કરવા બદલ આભાર.",
    "We've prepared the report for review.": "અમે સમીક્ષા માટે રિપોર્ટ તૈયાર કરી લીધો છે.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "તમે પછીના સંપર્ક માટે સંમતિ આપી હોય, તો જરૂર પડે તો યોગ્ય ટીમ "
        "તમારો સંપર્ક કરી શકે."
    ),
    "Is that right?": "શું તે બરાબર છે?",
    "Have I got that right?": "શું મેં સાચું સમજ્યું?",
    "Have I recorded those correctly?": "શું મેં તેમને બરાબર નોંધ્યા છે?",
}

_GUJARATI_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "સરનામું {0} તરીકે વંચાયું છે.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "સરનામું હજી પણ {0} જ છે.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "મેં તમારું ઈમેલ સરનામું {0} સમજ્યું.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "મેં તમારું ઈમેલ સરનામું {0} સાંભળ્યું.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "મને {0} સાંભળ્યું."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "મારી પાસે તમારું નામ {0} છે.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "મેં તમારી વિગતો આ રીતે નોંધી છે: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "મેં બરાબર નોંધ્યું છે કે નહીં તે ખાતરી કરવા હું પાછું વાંચીને સંભળાવું છું: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "મને {0} આંકડાનો નંબર મળ્યો છે, જેનો અંતિમ ભાગ {1} છે.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "માફ કરશો, મને સમજાયું નહીં - કૃપા કરીને સાચું {0} કહો.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "શું તમે {0} વિશે થોડું વધુ કહી શકશો?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "તમારી જન્મ તારીખ ({0}) અને તમારી ઉંમર ({1} વર્ષ) મેળ ખાતી નથી.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "હું કઈ બાબત નોંધું - તમારી જન્મ તારીખ કે તમારી ઉંમર?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0} માટે, કૃપા કરીને દવાના નિર્માતા અથવા તમારી સ્થાનિક "
        "પ્રતિકૂળ ઘટના રિપોર્ટિંગ ચેનલનો સંપર્ક કરો.",
    ),
)

# ---- Kannada (kn-IN) ----------------------------------------------------------

_KANNADA_SENTENCES: dict[str, str] = {
    "Thanks for that.": "ಇದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "Got it, thank you.": "ಅರ್ಥವಾಯಿತು, ಧನ್ಯವಾದಗಳು.",
    "Okay, thank you.": "ಸರಿ, ಧನ್ಯವಾದಗಳು.",
    "Thanks for the details.": "ವಿವರಗಳನ್ನು ತಿಳಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "ಇದನ್ನು ಹೇಳಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು, ನೀವು ಇದನ್ನು ಎದುರಿಸುತ್ತಿರುವುದನ್ನು ಕಂಡು ದುಃಖಿತರಾಗಿದ್ದೇನೆ."
    ),
    "I'll take a few details so we can record it safely.": (
        "ನಾವು ಇದನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ದಾಖಲಿಸಲು ಕೆಲವು ವಿವರಗಳನ್ನು ಪಡೆಯುತ್ತೇನೆ."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "ಮುಖ್ಯ: ತಿಳಿಸಿದ ಅಡ್ಡಪರಿಣಾಮಗಳ ಮೌಲ್ಯಮಾಪನ ಮತ್ತು ಚಿಕಿತ್ಸೆಗಾಗಿ ದಯವಿಟ್ಟು "
        "ಅರ್ಹ ವೈದ್ಯರು ಅಥವಾ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "ವೈದ್ಯಕೀಯ ಸಲಹೆ ಇಲ್ಲದೆ ಸ್ವತಃ ರೋಗನಿರ್ಣಯ ಮಾಡಬೇಡಿ ಅಥವಾ ನಿಮ್ಮ ಔಷಧ ಬದಲಾಯಿಸಬೇಡಿ."
    ),
    "Which medicine or product do you think may have caused this?": (
        "ಯಾವ ಔಷಧ ಅಥವಾ ಉತ್ಪನ್ನದಿಂದ ಇದಾಯಿತು ಎಂದು ನಿಮಗನಿಸುತ್ತದೆ?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "ನೀವು ಗಮನಿಸಿದ ಅಡ್ಡಪರಿಣಾಮ ಅಥವಾ ಪ್ರತಿಕ್ರಿಯೆ ಏನೆಂದು ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "ಇದನ್ನು ಅನುಭವಿಸಿದವರು ನೀವೇ, ಅಥವಾ ಬೇರೊಬ್ಬರ ಪರವಾಗಿ ತಿಳಿಸುತ್ತಿದ್ದೀರಾ?"
    ),
    "What is the patient's full name?": "ರೋಗಿಯ ಪೂರ್ಣ ಹೆಸರು ಎನು?",
    "May I have your first and last name?": "ನಿಮ್ಮ ಮೊದಲ ಮತ್ತು ಕೊನೆಯ ಹೆಸರು ತಿಳಿಯಬಹುದೇ?",
    "How old is the patient, or what is their date of birth?": (
        "ರೋಗಿಯ ವಯಸ್ಸು ಎಷ್ಟು, ಅಥವಾ ಅವರ ಜನ್ಮ ದಿನಾಂಕ ಎನು?"
    ),
    "How old are you, or what is your date of birth?": (
        "ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು, ಅಥವಾ ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕ ಎನು?"
    ),
    "And what is your name?": "ನಿಮ್ಮ ಹೆಸರು ಎನು?",
    "If we need a little more information later, is it alright if we contact you?": (
        "ನಂತರ ನಮಗೆ ಸ್ವಲ್ಪ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಬೇಕಾದರೆ, ನಾವು ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಲು ಸಾಧ್ಯವೇ?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "ನೀವು ದೂರವಾಣಿಯಿಂದಲೋ ಇಮೇಲ್‌ನಿಂದಲೋ ಸಂಪರ್ಕ ಬಯಸುತ್ತೀರಾ?"
    ),
    "What phone number would you like us to use?": "ನಾವು ಯಾವ ದೂರವಾಣಿ ಸಂಖ್ಯೆಯನ್ನು ಬಳಸಬೇಕು?",
    "What email address would you like us to use?": "ನಾವು ಯಾವ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ಬಳಸಬೇಕು?",
    "When did the side effect first start?": "ಅಡ್ಡಪರಿಣಾಮ ಮೊದಲು ಯಾವಾಗ ಆರಂಭವಾಯಿತು?",
    "Would you consider this problem serious?": "ಈ ಸಮಸ್ಯೆಯನ್ನು ನೀವು ಗಂಭೀರವೆಂದು ಪರಿಗಣಿಸುತ್ತೀರಾ?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "ರೋಗಿ ಈಗ ಹೇಗಿದ್ದಾರೆ - ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗಡಿದಿವೆಯೇ?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "ನೀವು ಈಗ ಹೇಗಿದ್ದೀರಿ - ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗಡಿದಿವೆಯೇ?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ - ಅದು ಸರಿಯಾಗಿದ್ದರೆ ಹೌದು ಎಂದು, ಬದಲಾಯಿಸಬೇಕಾದರೆ ಇಲ್ಲ ಎಂದು ಹೇಳಿ."
    ),
    "Is that correct?": "ಅದು ಸರಿಯೇ?",
    "Thank you - I've recorded everything we need for now.": (
        "ಧನ್ಯವಾದಗಳು - ಈಗಿನ ಸಲುವಾಗಿ ಬೇಕಾದಷ್ಟನ್ನು ನಾನು ದಾಖಲಿಸಿಕೊಂಡಿದ್ದೇನೆ."
    ),
    "A member of our team will follow up if needed.": (
        "ಅಗತ್ಯವಾದರೆ ನಮ್ಮ ತಂಡದ ಸದಸ್ಯರೊಬ್ಬರು ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಾರೆ."
    ),
    "Take care.": "ನಿಮ್ಮನ್ನು ನೀವು ನೋಡಿಕೊಳ್ಳಿ.",
    "Thank you.": "ಧನ್ಯವಾದಗಳು.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "ಮುಗಿಸುವ ಮೊದಲು, ಏನಾಯಿತು ಎಂಬುದರ ಬಗ್ಗೆ ಬೇರೆ ಏನಾದರೂ ಹಂಚಿಕೊಳ್ಳಬೇಕಿದೆಯೇ?"
    ),
    "Thank you for sharing that.": "ಅದನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "ಇದು ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಾಗಿದ್ದರೆ, ದಯವಿಟ್ಟು ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ನೆರವು ಪಡಿ "
        "ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ ತುರ್ತು ಸೇವೆಗೆ ಕರೆ ಮಾಡಿ."
    ),
    "This appears potentially urgent.": "ಇದು ಸಂಭಾವ್ಯವಾಗಿ ತುರ್ತು ಎನಿಸುತ್ತದೆ.",
    "A healthcare professional should be contacted immediately.": (
        "ತಕ್ಷಣ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಬೇಕು."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "ನಾನು ವೈಯಕ್ತಿಕ ಡೋಸೇಜ್, ಚಿಕಿತ್ಸಾ ಶಿಫಾರಸುಗಳು ಅಥವಾ ಔಷಧ ಬದಲಾವಣೆಗಳನ್ನು ತಿಳಿಸಲಾರೆ."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "ನಿಮ್ಮ ಔಷಧ ಪದ್ಧತಿಯಲ್ಲಿನ ಯಾವುದೇ ಬದಲಾವಣೆಗಾಗಿ ದಯವಿಟ್ಟು ಅರ್ಹ "
        "ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "I'm not sure how to interpret that.": "ಅದನ್ನು ಹೇಗೆ ಅರ್ಥೈಸಬೇಕೆಂದು ನನಗೆ ಖಚಿತವಿಲ್ಲ.",
    "Could you please clarify?": "ದಯವಿಟ್ಟು ನೀವು ಸ್ಪಷ್ಟಪಡಿಸಬಲ್ಲಿರಾ?",
    "Sorry, I didn't catch the product name.": "ಕ್ಷಮಿಸಿ, ಉತ್ಪನ್ನದ ಹೆಸರು ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ.",
    "Which medicine were you using?": "ನೀವು ಯಾವ ಔಷಧ ಬಳಸುತ್ತಿದ್ದಿರಿ?",
    "Sorry, could you describe the side effect in a few words?": (
        "ಕ್ಷಮಿಸಿ, ಅಡ್ಡಪರಿಣಾಮವನ್ನು ಕೆಲವು ಪದಗಳಲ್ಲಿ ವಿವರಿಸಬಲ್ಲಿರಾ?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು - ನೀವು ನಿಮ್ಮದೇ ಅನುಭವವನ್ನೋ ಬೇರೊಬ್ಬರದ್ದನ್ನೋ ತಿಳಿಸುತ್ತಿದ್ದೀರಾ?"
    ),
    "Could you say your first and last name again?": (
        "ನಿಮ್ಮ ಮೊದಲ ಮತ್ತು ಕೊನೆಯ ಹೆಸರನ್ನು ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "How old are you, in years?": "ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು, ವರ್ಷಗಳಲ್ಲಿ?",
    "Could you tell me your age in years, or your date of birth?": (
        "ವರ್ಷಗಳಲ್ಲಿ ನಿಮ್ಮ ವಯಸ್ಸು ಅಥವಾ ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you say the date of birth again?": "ಜನ್ಮ ದಿನಾಂಕವನ್ನು ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?",
    "Would it be alright if we contact you later - yes or no?": (
        "ನಾವು ನಂತರ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಿದರೆ ಸರಿಯೇ - ಹೌದೋ ಇಲ್ಲವೋ?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "ನಿಮಗೆ ಯಾವುದು ಸುಲಭ - ದೂರವಾಣಿ, ಇಮೇಲ್, ಅಥವಾ ಎರಡೂ?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "ದೂರವಾಣಿ ಸಂಖ್ಯೆಯನ್ನು ಅಂಕವಾರು ಅಂಕವಾಗಿ ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you repeat the email address?": "ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "ಅದು ಸುಮಾರು ಯಾವಾಗ ಆರಂಭವಾಯಿತು - ಉದಾಹರಣೆಗೆ ಇಂದು, ನಿನ್ನೆ, ಅಥವಾ ಕೆಲವು ದಿನಗಳ ಹಿಂದೆ?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "ಇದನ್ನು ನೀವು ಗಂಭೀರವೆಂದು ಪರಿಗಣಿಸುತ್ತೀರಾ - ಹೌದು, ಇಲ್ಲ, ಅಥವಾ ಖಚಿತವಿಲ್ಲ?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಈಗ ಸುಧಾರಿಸಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗಡಿದಿವೆಯೇ?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "ಇದಕ್ಕಾಗಿ ನೀವು ಯಾವುದೇ ವೈದ್ಯಕೀಯ ಆರೈಕೆ ಪಡೆದಿದ್ದೀರಾ - ಹೌದೋ ಇಲ್ಲವೋ?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "ರೋಗಿ ಈಗ ಹೇಗಿದ್ದಾರೆ - ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗಡಿದಿವೆಯೇ?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "ನೀವು ಈಗ ಹೇಗಿದ್ದೀರಿ - ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗಡಿದಿವೆಯೇ?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "ನೀವು ದೂರವಾಣಿಯಿಂದ, ಇಮೇಲ್‌ನಿಂದ, ಅಥವಾ ಎರಡರಿಂದಲೂ ಸಂಪರ್ಕ ಬಯಸುತ್ತೀರಾ?"
    ),
    "Thank you for sharing this information.": "ಈ ಮಾಹಿತಿಯನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "We've recorded what you've provided.": "ನೀವು ನೀಡಿದ ಮಾಹಿತಿಯನ್ನು ನಾವು ದಾಖಲಿಸಿಕೊಂಡಿದ್ದೇವೆ.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "ನೀವು ಮುಂದಿನ ಸಂಪರ್ಕಕ್ಕೆ ಸಮ್ಮತಿಸದ ಕಾರಣ, ಈ ವಿವರಗಳನ್ನು ಬಳಸಿ ನಾವು "
        "ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುವುದಿಲ್ಲ."
    ),
    "Thank you for your patience.": "ನಿಮ್ಮ ತಾಳ್ಮೆಗಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "ಕೆಲವು ವಿವರಗಳನ್ನು ನಾವು ದೃಢೀಕರಿಸಲಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ನಿಮ್ಮ ಪ್ರಕರಣವನ್ನು "
        "ನಮ್ಮ ತಂಡದ ಸದಸ್ಯರೊಬ್ಬರು ಪರಿಶೀಲಿಸುತ್ತಾರೆ."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "ಅದು ಸಂಪೂರ್ಣ ವಿಳಾಸವಾಗಿದ್ದರೆ ಹೌದು ಎಂದು ಹೇಳಿ - ಇನ್ನಷ್ಟು ಅಕ್ಷರಗಳಿದ್ದರೆ ದಯವಿಟ್ಟು ವಿಳಾಸವನ್ನು ಮತ್ತೆ ಹೇಳಿ."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "ಅದು ಸರಿಯಾಗಿಲ್ಲದಿದ್ದರೆ, ದಯವಿಟ್ಟು ಅದನ್ನು ಮತ್ತೆ ಒಂದೊಂದಾಗಿ ಅಕ್ಷರ ಅಥವಾ ಸಂಖ್ಯೆಯಾಗಿ ಹೇಳಿ."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "ನಮಗೆ ಹೆಸರನ್ನು ದೃಢೀಕರಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
    ),
    "We could not confirm the phone number.": (
        "ನಮಗೆ ಫೋನ್ ಸಂಖ್ಯೆಯನ್ನು ದೃಢೀಕರಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
    ),
    "We could not confirm the contact number.": (
        "ನಮಗೆ ಸಂಪರ್ಕ ಸಂಖ್ಯೆಯನ್ನು ದೃಢೀಕರಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "ನೀವು ಈಗ ಅದನ್ನು ಮತ್ತೆ ಹೇಳಬಹುದು, ಅಥವಾ ಪರಿಶೀಲನೆಯ ಸಮಯದಲ್ಲಿ ನಮ್ಮ ತಂಡ ಅದನ್ನು ಸಂಗ್ರಹಿಸುತ್ತದೆ."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "ಈ MedVaani ಪ್ರೋಟೋಟೈಪ್ ಕೇವಲ Lantus SoloStar ಗಾಗಿ ಸಿದ್ಧಪಡಿಸಲಾಗಿದೆ."
    ),
    "This prototype cannot continue this report.": "ಈ ಪ್ರೋಟೋಟೈಪ್ ಈ ವರದಿಯನ್ನು ಮುಂದುವರಿಸಲಾರದು.",
    "Thank you for taking the time to speak with us.": (
        "ನಮ್ಮೊಂದಿಗೆ ಮಾತನಾಡಲು ಸಮಯ ವಿನಿಯೋಗಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು."
    ),
    "I'm here to help record the event for review.": (
        "ಪರಿಶೀಲನೆಗಾಗಿ ಘಟನೆಯನ್ನು ದಾಖಲಿಸಲು ಸಹಾಯ ಮಾಡಲು ನಾನು ಇಲ್ಲಿದ್ದೇನೆ."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "ನಿಮಗೆ ಕೆಲವು ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳುತ್ತೇನೆ, ಮತ್ತು ಪರಿಶೀಲನೆಗಾಗಿ "
        "ವರದಿಯೊಂದನ್ನು ಸಿದ್ಧಪಡಿಸುತ್ತೇವೆ."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "ನಾನು ಆರೋಗ್ಯ ವೃತ್ತಿಪರರ ಬದಲಾಗಿಲ್ಲ, ಮತ್ತು ವೈದ್ಯಕೀಯ ಸಲಹೆ ನೀಡಲಾರೆ."
    ),
    "Please tell me what happened, in your own words.": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಮಾತುಗಳಲ್ಲಿ ಏನಾಯಿತು ಎಂದು ಹೇಳಿ.",
    "Take your time\u2014I'm listening.": "ನಿಮ್ಮ ಸಮಯ ತೆಗೆದುಕೊಳ್ಳಿ\u2014ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ.",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "ನಿಮ್ಮ ವರದಿಯನ್ನು ಸರಿಯಾಗಿ ದಾಖಲಿಸಲು ಸಹಾಯವಾಗಿ, ಸಂಬಂಧಿಸಿದ ಔಷಧದ "
        "ಹೆಸರು ಎನೆಂದು ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "I'm sorry to hear that you've experienced this.": "ನೀವು ಇದನ್ನು ಅನುಭವಿಸಿದ್ದನ್ನು ಕೇಳಿ ದುಃಖಿತರಾಗಿದ್ದೇನೆ.",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "ಸರಿಯಾದ ಔಷಧ ನನ್ನ ಬಳಿ ಇದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು, ಅದು Lantus SoloStarನೇ?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "ಈ ಘಟನೆಯನ್ನು ಅನುಭವಿಸಿದವರು ನೀವೇ, ಅಥವಾ ಬೇರೊಬ್ಬರ ಪರವಾಗಿ "
        "ತಿಳಿಸುತ್ತಿದ್ದೀರಾ ಎಂದು ಕೇಳಬಹುದೇ?"
    ),
    "Could you please tell me your full name?": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರನ್ನು ಹೇಳಬಲ್ಲಿರಾ?",
    "Could you please share your contact number?": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸಂಪರ್ಕ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಬಲ್ಲಿರಾ?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "ದಯವಿಟ್ಟು ರೋಗಿಯ ಮೊದಲ ಮತ್ತು ಕೊನೆಯ ಹೆಸರು, ಜೊತೆಗೆ ಅವರ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಮೊದಲ ಮತ್ತು ಕೊನೆಯ ಹೆಸರು, ಜೊತೆಗೆ ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you please tell me the patient's date of birth?": "ದಯವಿಟ್ಟು ರೋಗಿಯ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಬಲ್ಲಿರಾ?",
    "Could you please tell me your date of birth?": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಬಲ್ಲಿರಾ?",
    "Was it the name or the date of birth that was incorrect?": "ಹೆಸರು ತಪ್ಪಾಗಿತ್ತೋ ಅಥವಾ ಜನ್ಮ ದಿನಾಂಕವೋ?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "ಕ್ಷಮಿಸಿ, ಹೆಸರೋ ಅಥವಾ ಜನ್ಮ ದಿನಾಂಕವೋ ಯಾವುದು ತಪ್ಪಾಗಿತ್ತು ಎಂದು ಹೇಳಿ."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು - ಅದು ಬೇರೊಂದು ಲಕ್ಷಣವಾಗಿತ್ತೋ, ಅಥವಾ ನೀವು "
        "ವಿವರಿಸಿದ್ದಕ್ಕೆ ತಿದ್ದುಪಡಿಯಾಗಿತ್ತೋ?"
    ),
    "Sorry, please tell me a full date of birth.": "ಕ್ಷಮಿಸಿ, ದಯವಿಟ್ಟು ಪೂರ್ಣ ಜನ್ಮ ದಿನಾಂಕ ಹೇಳಿ.",
    "Sorry, I couldn't catch that.": (
        "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "ನನಗೆ ತಿಂಗಳು ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿರಲಿಲ್ಲ - ದಯವಿಟ್ಟು ಹುಟ್ಟಿದ ದಿನಾಂಕವನ್ನು ದಿನ, ತಿಂಗಳು ಮತ್ತು ವರ್ಷವಾಗಿ ಹೇಳಿ."
    ),
    "Could you please repeat it?": (
        "ದಯವಿಟ್ಟು ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "I couldn't catch the email address.": (
        "ಇಮೇಲ್ ವಿಳಾಸ ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ."
    ),
    "Could you please share the patient's contact number?": (
        "ದಯವಿಟ್ಟು ರೋಗಿಯ ಸಂಪರ್ಕ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you please share the best contact number for you?": (
        "ದಯವಿಟ್ಟು ನಿಮಗೆ ಸೂಕ್ತವಾದ ಸಂಪರ್ಕ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "ಸಂಪರ್ಕ ಸಂಖ್ಯೆಯನ್ನು ಅಂಕವಾರು ಅಂಕವಾಗಿ ಮತ್ತೆ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "ಮುಂದೆ ಯಾವುದೇ ಮಾಹಿತಿ ಬೇಕಾದರೆ, ಸೂಕ್ತ ತಂಡವು ಈ ವರದಿ ಕುರಿತು "
        "ರೋಗಿಯನ್ನು ಸಂಪರ್ಕಿಸಿದರೆ ಸರಿಯೇ?"
    ),
    "May I have the patient's email address, if available?": (
        "ಲಭ್ಯವಿದ್ದರೆ, ರೋಗಿಯ ಇಮೇಲ್ ವಿಳಾಸ ತಿಳಿಯಬಹುದೇ?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "ದಯವಿಟ್ಟು ನೀವು ದಾಖಲಿಸಬೇಕೆಂಬ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ಹೇಳಿ."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ಒಂದೊಂದಾಗಿ ಅಕ್ಷರ ಅಥವಾ ಸಂಖ್ಯೆಯಾಗಿ ಹೇಳಬಲ್ಲಿರಾ?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "@ ಗಾಗಿ 'at the rate' ಮತ್ತು ತಿಳಿಹಂದಿಗಾಗಿ 'dot' ಎಂದು ಹೇಳಬಹುದು."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "ಈ ಘಟನೆ ಗಂಭೀರ ಅಥವಾ ಜೀವಕ್ಕೆ ಅಪಾಯಕಾರಿಯಾಗಿತ್ತು ಎಂದು ನೀವು ಹೇಳುತ್ತೀರಾ?"
    ),
    "Please seek immediate medical attention now.": "ದಯವಿಟ್ಟು ಈಗಿನಿಂದ ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ನೆರವು ಪಡಿ.",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "ನೀವು ಈಗಾಗಲೇ ಮುಂದಿನ ಸಂಪರ್ಕಕ್ಕೆ ಸಮ್ಮತಿಸಿದ್ದರೆ, ಹೆಚ್ಚುವರಿ ಮಾಹಿತಿಯನ್ನು "
        "ಹೆಚ್ಚು ಅನುಕೂಲಕರ ಸಮಯದಲ್ಲಿ ಸಂಗ್ರಹಿಸುವ ವ್ಯವಸ್ಥೆ ಮಾಡುತ್ತೇವೆ."
    ),
    "Could you tell me when the event occurred?": "ಘಟನೆ ಯಾವಾಗ ನಡೆಯಿತು ಎಂದು ಹೇಳಬಲ್ಲಿರಾ?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "ಘಟನೆಯನ್ನು ದಾಖಲಿಸಲು ಸಹಾಯವಾಗಬಲ್ಲ ಬೇರೆ ಏನಾದರೂ ಹಂಚಿಕೊಳ್ಳಬೇಕಿದೆಯೇ?"
    ),
    "Of course.": "ಖಂಡಿತ.",
    "Please tell me any additional details you'd like included.": (
        "ದಯವಿಟ್ಟು ನೀವು ಸೇರಿಸಬೇಕೆಂಬ ಹೆಚ್ಚುವರಿ ವಿವರಗಳನ್ನು ಹೇಳಿ."
    ),
    "Thank you for sharing this information with us.": "ನಮ್ಮೊಂದಿಗೆ ಈ ಮಾಹಿತಿಯನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
    "We've prepared the report for review.": "ನಾವು ಪರಿಶೀಲನೆಗಾಗಿ ವರದಿಯನ್ನು ಸಿದ್ಧಪಡಿಸಿದ್ದೇವೆ.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "ನೀವು ಮುಂದಿನ ಸಂಪರ್ಕಕ್ಕೆ ಸಮ್ಮತಿ ನೀಡಿದ್ದರೆ, ಅಗತ್ಯವಾದರೆ ಸೂಕ್ತ ತಂಡವು "
        "ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಬಹುದು."
    ),
    "Is that right?": "ಅದು ಸರಿಯೇ?",
    "Have I got that right?": "ನಾನು ಸರಿಯಾಗಿ ಅರ್ಥ ಮಾಡಿಕೊಂಡಿದ್ದೇನೆಯೇ?",
    "Have I recorded those correctly?": "ಅವುಗಳನ್ನು ನಾನು ಸರಿಯಾಗಿ ದಾಖಲಿಸಿದ್ದೇನೆಯೇ?",
}

_KANNADA_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "ವಿಳಾಸವು {0} ಆಗಿ ಓದಲ್ಪಟ್ಟಿದೆ.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "ವಿಳಾಸವು ಈಗಲೂ {0} ಆಗಿಯೇ ಇದೆ.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ನಾನು {0} ಎಂದು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ನಾನು {0} ಎಂದು ಕೇಳಿದೆ.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "ನಾನು {0} ಕೇಳಿದೆ."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "ನಿಮ್ಮ ಹೆಸರು {0} ಎಂದು ನನ್ನ ಬಳಿ ದಾಖಲಾಗಿದೆ.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "ನಿಮ್ಮ ವಿವರಗಳನ್ನು ನಾನು ಹೀಗೆ ದಾಖಲಿಸಿದ್ದೇನೆ: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "ಸರಿಯಾಗಿ ದಾಖಲಿಸಿದ್ದೇನೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ನಾನು ಮತ್ತೆ ಓದಿ ತೋರಿಸುತ್ತೇನೆ: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "ನನಗೆ {0} ಅಂಕಿಗಳ ಸಂಖ್ಯೆ ದೊರೆತಿದೆ, ಅದರ ಕೊನೆಯ ಭಾಗ {1} ಆಗಿದೆ.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ - ದಯವಿಟ್ಟು ಸರಿಯಾದ {0} ಹೇಳಿ.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "{0} ಬಗ್ಗೆ ಸ್ವಲ್ಪ ಹೆಚ್ಚು ಹೇಳಬಲ್ಲಿರಾ?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕ ({0}) ಮತ್ತು ನಿಮ್ಮ ವಯಸ್ಸು ({1} ವರ್ಷಗಳು) ಹೊಂದಾಣಿಕೆಯಾಗುತ್ತಿಲ್ಲ.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "ಯಾವುದನ್ನು ದಾಖಲಿಸಬೇಕು - ನಿಮ್ಮ ಜನ್ಮ ದಿನಾಂಕವೋ ನಿಮ್ಮ ವಯಸ್ಸೋ?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0} ಗಾಗಿ, ದಯವಿಟ್ಟು ಔಷಧದ ತಯಾರಕರನ್ನು ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ "
        "ಪ್ರತಿಕೂಲ ಘಟನೆ ವರದಿ ಮಾಡುವ ಮಾಧ್ಯಮವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    ),
)

# ---- Malayalam (ml-IN) --------------------------------------------------------

_MALAYALAM_SENTENCES: dict[str, str] = {
    "Thanks for that.": "അതിന് നന്ദി.",
    "Got it, thank you.": "മനസ്സിലായി, നന്ദി.",
    "Okay, thank you.": "ശരി, നന്ദി.",
    "Thanks for the details.": "വിവരങ്ങൾ നൽകിയതിന് നന്ദി.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "എന്നോട് പറഞ്ഞതിന് നന്ദി; ഇത് അനുഭവിക്കേണ്ടി വന്നതിൽ എനിക്ക് ഖേദമുണ്ട്."
    ),
    "I'll take a few details so we can record it safely.": (
        "ഇത് സുരക്ഷിതമായി രേഖപ്പെടുത്താൻ ഞാൻ ചില വിവരങ്ങൾ എടുക്കാം."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "പ്രധാനം: റിപ്പോർട്ട് ചെയ്ത പാർശ്വഫലങ്ങളുടെ വിലയിരുത്തലിനും ചികിത്സയ്ക്കും "
        "ദയവായി യോഗ്യതയുള്ള ഒരു ഡോക്ടറെയോ ആരോഗ്യ പ്രവർത്തകനെയോ സമീപിക്കുക."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "വൈദ്യസലഹമില്ലാതെ സ്വയം രോഗനിർണയം നടത്തരുത്, മരുന്ന് മാറ്റരുത്."
    ),
    "Which medicine or product do you think may have caused this?": (
        "ഏത് മരുന്നോ ഉൽപ്പന്നമോ ആണ് ഇതിന് കാരണമായതെന്ന് നിങ്ങൾക്ക് തോന്നുന്നത്?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "നിങ്ങൾ ശ്രദ്ധിച്ച പാർശ്വഫലം അല്ലെങ്കിൽ പ്രതിക്രിയ എന്താണെന്ന് പറയാമോ?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "ഇത് അനുഭവിച്ചത് നിങ്ങളാണോ, അതോ മറ്റൊരാൾക്ക് വേണ്ടിയാണോ റിപ്പോർട്ട് ചെയ്യുന്നത്?"
    ),
    "What is the patient's full name?": "രോഗിയുടെ പൂർണ്ണ പേര് എന്താണ്?",
    "May I have your first and last name?": "നിങ്ങളുടെ പേരും കുടുംബപ്പേരും അറിയാമോ?",
    "How old is the patient, or what is their date of birth?": (
        "രോഗിക്ക് പ്രായം എത്രയാണ്, അല്ലെങ്കിൽ ജനനത്തീയതി എന്താണ്?"
    ),
    "How old are you, or what is your date of birth?": (
        "നിങ്ങൾക്ക് പ്രായം എത്രയാണ്, അല്ലെങ്കിൽ ജനനത്തീയതി എന്താണ്?"
    ),
    "And what is your name?": "നിങ്ങളുടെ പേര് എന്താണ്?",
    "If we need a little more information later, is it alright if we contact you?": (
        "പിന്നീട് കുറച്ചുകൂടി വിവരങ്ങൾ വേണമെങ്കിൽ, നിങ്ങളെ ബന്ധപ്പെട്ടാൽ ശരിയാകുമോ?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "ഫോൺ വഴിയോ ഇമെയിൽ വഴിയോ ആണോ ബന്ധപ്പെടാൻ ഇഷ്ടപ്പെടുന്നത്?"
    ),
    "What phone number would you like us to use?": "ഏത് ഫോൺ നമ്പർ ഉപയോഗിക്കണം?",
    "What email address would you like us to use?": "ഏത് ഇമെയിൽ വിലാസം ഉപയോഗിക്കണം?",
    "When did the side effect first start?": "പാർശ്വഫലം ആദ്യമായി എപ്പോൾ ആരംഭിച്ചു?",
    "Would you consider this problem serious?": "ഈ പ്രശ്നം ഗൗരവമുള്ളതാണെന്ന് നിങ്ങൾ കരുതുന്നുണ്ടോ?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "രോഗിക്ക് ഇപ്പോൾ എങ്ങനെ തോന്നുന്നു - ലക്ഷണങ്ങൾ മെച്ചപ്പെട്ടോ, അതേപടിയോ, അതോ വഷളായോ?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "നിങ്ങൾക്ക് ഇപ്പോൾ എങ്ങനെ തോന്നുന്നു - ലക്ഷണങ്ങൾ മെച്ചപ്പെട്ടോ, അതേപടിയോ, അതോ വഷളായോ?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല - ശരിയാണെങ്കിൽ അതെ എന്നും, മാറ്റണമെങ്കിൽ ഇല്ല എന്നും പറയൂ."
    ),
    "Is that correct?": "അത് ശരിയാണോ?",
    "Thank you - I've recorded everything we need for now.": (
        "നന്ദി - ഇപ്പോൾ ആവശ്യമുള്ളതെല്ലാം ഞാൻ രേഖപ്പെടുത്തിക്കഴിഞ്ഞു."
    ),
    "A member of our team will follow up if needed.": (
        "ആവശ്യമെങ്കിൽ ഞങ്ങളുടെ ടീമിലെ ഒരാൾ ബന്ധപ്പെടും."
    ),
    "Take care.": "നിങ്ങൾ ശ്രദ്ധിക്കുക.",
    "Thank you.": "നന്ദി.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "തീരുന്നതിന് മുമ്പ്, എന്ത് സംഭവിച്ചു എന്നതിനെക്കുറിച്ച് മറ്റെന്തെങ്കിലും പറയാനുണ്ടോ?"
    ),
    "Thank you for sharing that.": "അത് പങ്കിട്ടതിന് നന്ദി.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "ഇത് ഒരു മെഡിക്കൽ അത്യാഹിതമെങ്കിൽ, ദയവായി ഉടനടി വൈദ്യസഹായം തേടുക "
        "അല്ലെങ്കിൽ നിങ്ങളുടെ പ്രാദേശിക അത്യാഹിത സേവനത്തെ വിളിക്കുക."
    ),
    "This appears potentially urgent.": "ഇത് സാധ്യതയുള്ള അത്യാഹിത സാഹചര്യമായി തോന്നുന്നു.",
    "A healthcare professional should be contacted immediately.": (
        "ഉടനടി ഒരു ആരോഗ്യ വിദഗ്ധരെ ബന്ധപ്പെടണം."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "വ്യക്തിഗത ഡോസും ചികിത്സാ നിർദ്ദേശങ്ങളും മരുന്ന് മാറ്റങ്ങളും ഞാൻ നൽകാൻ കഴിയില്ല."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "മരുന്നിന്റെ ക്രമത്തിൽ വരുന്ന മാറ്റങ്ങൾക്ക് ദയവായി യോഗ്യതയുള്ള "
        "ഒരു ആരോഗ്യ വിദഗ്ധരെ സമീപിക്കുക."
    ),
    "I'm not sure how to interpret that.": "അത് എങ്ങനെ മനസ്സിലാക്കണം എന്ന് എനിക്ക് ഉറപ്പില്ല.",
    "Could you please clarify?": "ദയവായി വ്യക്തമാക്കാമോ?",
    "Sorry, I didn't catch the product name.": "ക്ഷമിക്കണം, ഉൽപ്പന്നത്തിന്റെ പേര് എനിക്ക് മനസ്സിലായില്ല.",
    "Which medicine were you using?": "ഏത് മരുന്നാണ് ഉപയോഗിച്ചിരുന്നത്?",
    "Sorry, could you describe the side effect in a few words?": (
        "ക്ഷമിക്കണം, പാർശ്വഫലം കുറച്ചുവാക്കുകളിൽ വിവരിക്കാമോ?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "ഉറപ്പാക്കാൻ - നിങ്ങളുടെ സ്വന്തം അനുഭവമാണോ പറയുന്നത്, അതോ മറ്റാരുടേതോ?"
    ),
    "Could you say your first and last name again?": (
        "നിങ്ങളുടെ പേരും കുടുംബപ്പേരും വീണ്ടും പറയാമോ?"
    ),
    "How old are you, in years?": "പ്രായം എത്രയാണ്, വർഷങ്ങളിൽ?",
    "Could you tell me your age in years, or your date of birth?": (
        "വർഷങ്ങളിലായി പ്രായമോ ജനനത്തീയതിയോ പറയാമോ?"
    ),
    "Could you say the date of birth again?": "ജനനത്തീയതി വീണ്ടും പറയാമോ?",
    "Would it be alright if we contact you later - yes or no?": (
        "പിന്നീട് നിങ്ങളെ ബന്ധപ്പെട്ടാൽ ശരിയാകുമോ - അതെ അല്ലെങ്കിൽ ഇല്ല?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "ഏതാണ് നിങ്ങൾക്ക് എളുപ്പം - ഫോൺ, ഇമെയിൽ, അതോ രണ്ടും?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "ഫോൺ നമ്പർ ഓരോ അക്കമായി വീണ്ടും പറയാമോ?"
    ),
    "Could you repeat the email address?": "ഇമെയിൽ വിലാസം വീണ്ടും പറയാമോ?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "ഏകദേശം എപ്പോഴാണ് തുടങ്ങിയത് - ഉദാഹരണത്തിന് ഇന്ന്, ഇന്നലെ, അതോ കുറച്ച് ദിവസങ്ങൾക്ക് മുമ്പോ?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "ഇത് ഗൗരവമാണെന്ന് നിങ്ങൾ കരുതുന്നോ - അതെ, ഇല്ല, അതോ ഉറപ്പില്ലേ?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "ലക്ഷണങ്ങൾ ഇപ്പോൾ മെച്ചപ്പെട്ടോ, അതേപടിയോ, അതോ വഷളായോ?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "അതിന് ഏതെങ്കിലും വൈദ്യസഹായം ലഭിച്ചോ - അതെ അല്ലെങ്കിൽ ഇല്ല?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "രോഗിക്ക് ഇപ്പോൾ എങ്ങനെ തോന്നുന്നു - ലക്ഷണങ്ങൾ മെച്ചപ്പെട്ടോ, അതേപടിയോ, അതോ വഷളായോ?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "നിങ്ങൾക്ക് ഇപ്പോൾ എങ്ങനെ തോന്നുന്നു - ലക്ഷണങ്ങൾ മെച്ചപ്പെട്ടോ, അതേപടിയോ, അതോ വഷളായോ?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "ഫോൺ വഴിയോ, ഇമെയിൽ വഴിയോ, അതോ രണ്ടും കൂടിയോ ആണോ ബന്ധപ്പെടാൻ ഇഷ്ടപ്പെടുന്നത്?"
    ),
    "Thank you for sharing this information.": "ഈ വിവരങ്ങൾ പങ്കിട്ടതിന് നന്ദി.",
    "We've recorded what you've provided.": "നിങ്ങൾ നൽകിയ വിവരങ്ങൾ ഞങ്ങൾ രേഖപ്പെടുത്തിയിട്ടുണ്ട്.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "നിങ്ങൾ പിന്തുടർച്ചാ ബന്ധത്തിന് സമ്മതിച്ചിട്ടില്ലാത്തതിനാൽ, ഈ വിവരങ്ങൾ "
        "ഉപയോഗിച്ച് ഞങ്ങൾ നിങ്ങളെ ബന്ധപ്പെടില്ല."
    ),
    "Thank you for your patience.": "നിങ്ങളുടെ ക്ഷമയ്ക്ക് നന്ദി.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "ചില വിവരങ്ങൾ സ്ഥിരീകരിക്കാൻ ഞങ്ങൾക്ക് കഴിഞ്ഞില്ല; അതിനാൽ നിങ്ങളുടെ "
        "കേസ് ഞങ്ങളുടെ ടീമിലെ ഒരാൾ പരിശോധിക്കും."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "അത് പൂർണ്ണമായ വിലാസമാണെങ്കിൽ അതെ എന്ന് പറയുക - കൂടുതൽ അക്ഷരങ്ങൾ ഉണ്ടെങ്കിൽ ദയവായി വിലാസം വീണ്ടും പറയുക."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "അത് ശരിയല്ലെങ്കിൽ, ദയവായി അത് വീണ്ടും ഓരോ അക്ഷരമോ അക്കമോ ആയി പറയുക."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "ഞങ്ങൾക്ക് പേര് സ്ഥിരീകരിക്കാൻ കഴിഞ്ഞില്ല."
    ),
    "We could not confirm the phone number.": (
        "ഞങ്ങൾക്ക് ഫോൺ നമ്പർ സ്ഥിരീകരിക്കാൻ കഴിഞ്ഞില്ല."
    ),
    "We could not confirm the contact number.": (
        "ഞങ്ങൾക്ക് കോൺടാക്റ്റ് നമ്പർ സ്ഥിരീകരിക്കാൻ കഴിഞ്ഞില്ല."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "നിങ്ങൾക്ക് ഇപ്പോൾ അത് വീണ്ടും പറയാം, അല്ലെങ്കിൽ സമീക്ഷയിൽ ഞങ്ങളുടെ ടീം അത് ശേഖരിക്കും."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "ഈ MedVaani പ്രോട്ടോടൈപ്പ് Lantus SoloStar-ന് മാത്രമായി ക്രമീകരിച്ചിരിക്കുന്നു."
    ),
    "This prototype cannot continue this report.": "ഈ പ്രോട്ടോടൈപ്പിന് ഈ റിപ്പോർട്ട് തുടരാൻ കഴിയില്ല.",
    "Thank you for taking the time to speak with us.": "ഞങ്ങളുമായി സംസാരിക്കാൻ സമയം ചെലവഴിച്ചതിന് നന്ദി.",
    "I'm here to help record the event for review.": (
        "പരിശോധനയ്ക്കായി സംഭവം രേഖപ്പെടുത്താൻ സഹായിക്കാനാണ് ഞാൻ ഇവിടെ."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "ഞാൻ ചില ചോദ്യങ്ങൾ ചോദിക്കും; പരിശോധനയ്ക്കായി ഒരു റിപ്പോർട്ട് "
        "ഞങ്ങൾ തയ്യാറാക്കും."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "ഞാൻ ഒരു ആരോഗ്യ വിദഗ്ധന് പകരമല്ല; വൈദ്യസലഹം നൽകാൻ എനിക്ക് കഴിയില്ല."
    ),
    "Please tell me what happened, in your own words.": "എന്ത് സംഭവിച്ചു എന്ന് നിങ്ങളുടെ സ്വന്തം വാക്കുകളിൽ പറയൂ.",
    "Take your time\u2014I'm listening.": "സമയമെടുക്കുക\u2014ഞാൻ കേൾക്കുകയാണ്.",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "റിപ്പോർട്ട് ശരിയായി രേഖപ്പെടുത്താൻ സഹായിക്കാൻ, ബന്ധപ്പെട്ട മരുന്നിന്റെ "
        "പേര് പറയാമോ?"
    ),
    "I'm sorry to hear that you've experienced this.": "നിങ്ങൾ ഇത് അനുഭവിച്ചു എന്നറിഞ്ഞപ്പോൾ ഖേദമുണ്ടായി.",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "ശരിയായ മരുന്നാണ് എന്ന് ഉറപ്പാക്കാൻ - അത് Lantus SoloStar ആണോ?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "ഈ സംഭവം അനുഭവിച്ചത് നിങ്ങളാണോ, അതോ മറ്റൊരാൾക്ക് വേണ്ടിയാണോ "
        "റിപ്പോർട്ട് ചെയ്യുന്നത് എന്ന് ചോദിക്കട്ടെ?"
    ),
    "Could you please tell me your full name?": "ദയവായി നിങ്ങളുടെ പൂർണ്ണ പേര് പറയാമോ?",
    "Could you please share your contact number?": "ദയവായി നിങ്ങളുടെ ബന്ധപ്പെടാനുള്ള നമ്പർ പറയാമോ?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "ദയവായി രോഗിയുടെ പേരും കുടുംബപ്പേരും, ജനനത്തീയതിയും പറയാമോ?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "ദയവായി നിങ്ങളുടെ പേരും കുടുംബപ്പേരും, ജനനത്തീയതിയും പറയാമോ?"
    ),
    "Could you please tell me the patient's date of birth?": "ദയവായി രോഗിയുടെ ജനനത്തീയതി പറയാമോ?",
    "Could you please tell me your date of birth?": "ദയവായി നിങ്ങളുടെ ജനനത്തീയതി പറയാമോ?",
    "Was it the name or the date of birth that was incorrect?": "പേരാണോ ജനനത്തീയതിയാണോ തെറ്റായത്?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "ക്ഷമിക്കണം, പേരോ ജനനത്തീയതിയോ ഏതാണ് തെറ്റായത് എന്ന് പറയൂ."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "ഉറപ്പാക്കാൻ - അത് വേറൊരു ലക്ഷണമായിരുന്നോ, അതോ നിങ്ങൾ "
        "വിവരിച്ചതിലുള്ള തിരുത്തലോ?"
    ),
    "Sorry, please tell me a full date of birth.": "ക്ഷമിക്കണം, ദയവായി പൂർണ്ണ ജനനത്തീയതി പറയൂ.",
    "Sorry, I couldn't catch that.": (
        "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "മാസം വ്യക്തമായി കേട്ടില്ല - ദയവായി ജന്മതീയതി ദിവസം, മാസം, വർഷം എന്നീ ക്രമത്തിൽ പറയുക."
    ),
    "Could you please repeat it?": (
        "ദയവായി വീണ്ടും പറയാമോ?"
    ),
    "I couldn't catch the email address.": (
        "ഇമെയിൽ വിലാസം എനിക്ക് മനസ്സിലായില്ല."
    ),
    "Could you please share the patient's contact number?": (
        "ദയവായി രോഗിയുടെ ബന്ധപ്പെടാനുള്ള നമ്പർ പറയാമോ?"
    ),
    "Could you please share the best contact number for you?": (
        "ദയവായി നിങ്ങൾക്ക് ഏറ്റവും സൗകര്യപ്രദമായ ബന്ധപ്പെടാനുള്ള നമ്പർ പറയാമോ?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "ബന്ധപ്പെടാനുള്ള നമ്പർ ഓരോ അക്കമായി വീണ്ടും പറയാമോ?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "കൂടുതൽ വിവരങ്ങൾ വേണമെങ്കിൽ, ബന്ധപ്പെട്ട ടീമിന് ഈ റിപ്പോർട്ടിനെക്കുറിച്ച് "
        "രോഗിയെ ബന്ധപ്പെടാൻ കഴിയുമോ?"
    ),
    "May I have the patient's email address, if available?": (
        "ലഭ്യമെങ്കിൽ, രോഗിയുടെ ഇമെയിൽ വിലാസം അറിയാമോ?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "രേഖപ്പെടുത്തേണ്ട ഇമെയിൽ വിലാസം ദയവായി പറയൂ."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "ഇമെയിൽ വിലാസം ഓരോ അക്ഷരമോ അക്കമോ ആയി പറയാമോ?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "'@'ന് 'at the rate' എന്നും പോയിന്റിന് 'dot' എന്നും പറയാം."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "സംഭവം ഗൗരവമുള്ളതോ ജീവന് ഭീഷണിയോ ആയിരുന്നു എന്ന് നിങ്ങൾ പറയുമോ?"
    ),
    "Please seek immediate medical attention now.": "ദയവായി ഉടനടി വൈദ്യസഹായം തേടുക.",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "നിങ്ങൾ ഇതിനകം പിന്തുടർച്ചാ ബന്ധത്തിന് സമ്മതിച്ചിട്ടുണ്ടെങ്കിൽ, കൂടുതൽ "
        "വിവരങ്ങൾ സൗകര്യപ്രദമായ സമയത്ത് ശേഖരിക്കാൻ ഞങ്ങൾ ഏർപ്പാട് ചെയ്യും."
    ),
    "Could you tell me when the event occurred?": "സംഭവം എപ്പോൾ ഉണ്ടായി എന്ന് പറയാമോ?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "സംഭവം രേഖപ്പെടുത്താൻ സഹായിക്കുന്ന മറ്റെന്തെങ്കിലും പങ്കിടാനുണ്ടോ?"
    ),
    "Of course.": "തീർച്ചയായും.",
    "Please tell me any additional details you'd like included.": (
        "ഉൾപ്പെടുത്താൻ ആഗ്രഹിക്കുന്ന കൂടുതൽ വിവരങ്ങൾ ദയവായി പറയൂ."
    ),
    "Thank you for sharing this information with us.": "ഈ വിവരങ്ങൾ ഞങ്ങളുമായി പങ്കിട്ടതിന് നന്ദി.",
    "We've prepared the report for review.": "പരിശോധനയ്ക്കായി റിപ്പോർട്ട് ഞങ്ങൾ തയ്യാറാക്കിയിട്ടുണ്ട്.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "നിങ്ങൾ പിന്തുടർച്ചാ ബന്ധത്തിന് സമ്മതി നൽകിയിട്ടുണ്ടെങ്കിൽ, ആവശ്യമെങ്കിൽ "
        "ബന്ധപ്പെട്ട ടീം നിങ്ങളെ ബന്ധപ്പെടാം."
    ),
    "Is that right?": "അത് ശരിയാണോ?",
    "Have I got that right?": "ഞാൻ ശരിയായി മനസ്സിലാക്കിയോ?",
    "Have I recorded those correctly?": "അവ ഞാൻ ശരിയായി രേഖപ്പെടുത്തിയോ?",
}

_MALAYALAM_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "വിലാസം {0} ആയി വായിച്ചു.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "വിലാസം ഇപ്പോഴും {0} തന്നെയാണ്.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "ഞാൻ നിങ്ങളുടെ ഇമെയിൽ വിലാസം {0} ആയി മനസ്സിലാക്കി.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "ഞാൻ നിങ്ങളുടെ ഇമെയിൽ വിലാസം {0} ആയി കേട്ടു.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "ഞാൻ {0} കേട്ടു."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "നിങ്ങളുടെ പേര് {0} എന്നാണ് എന്റെ അടുത്ത് രേഖപ്പെടുത്തിയിരിക്കുന്നത്.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "നിങ്ങളുടെ വിവരങ്ങൾ ഞാൻ ഇങ്ങനെ രേഖപ്പെടുത്തി: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "ശരിയായി രേഖപ്പെടുത്തിയോ എന്ന് ഉറപ്പാക്കാൻ വീണ്ടും വായിച്ചു കേൾപ്പിക്കാം: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "എനിക്ക് {0} അക്കത്തിലുള്ള നമ്പർ ലഭിച്ചു, അതിന്റെ അവസാന ഭാഗം {1} ആണ്.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല - ദയവായി ശരിയായ {0} പറയൂ.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "{0}-നെക്കുറിച്ച് കുറച്ചുകൂടി പറയാമോ?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "നിങ്ങളുടെ ജനനത്തീയതി ({0}) പ്രായം ({1} വർഷം) എന്നിവ ചേരുന്നില്ല.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "ഏതാണ് രേഖപ്പെടുത്തേണ്ടത് - ജനനത്തീയതിയോ പ്രായമോ?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0}-ന്, ദയവായി മരുന്നിന്റെ നിർമ്മാതാവിനെയോ നിങ്ങളുടെ സ്ഥാനിക "
        "പ്രതികൂല സംഭവ റിപ്പോർട്ടിംഗ് ചാനലിനെയോ ബന്ധപ്പെടുക.",
    ),
)

# ---- Marathi (mr-IN) ----------------------------------------------------------

_MARATHI_SENTENCES: dict[str, str] = {
    "Thanks for that.": "याबद्दल धन्यवाद.",
    "Got it, thank you.": "समजले, धन्यवाद.",
    "Okay, thank you.": "ठीक आहे, धन्यवाद.",
    "Thanks for the details.": "तपशील देण्याबद्दल धन्यवाद.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "मला सांगितल्याबद्दल धन्यवाद, आणि तुम्ही हे सर्व सहन करत आहात याबद्दल मला खेद आहे."
    ),
    "I'll take a few details so we can record it safely.": (
        "आपण तो सुरक्षितपणे नोंदवू शकू यासाठी मी काही तपशील घेईन."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "महत्त्वाचे: नोंदवलेल्या दुष्परिणामांचे मूल्यांकन आणि उपचारासाठी कृपया "
        "एका योग्य डॉक्टरांचा किंवा आरोग्य व्यावसायिकाचा सल्ला घ्या."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "वैद्यकीय सल्ल्याशिवाय स्वतःचे निदान करू नका किंवा तुमची औषधे बदलू नका."
    ),
    "Which medicine or product do you think may have caused this?": (
        "कोणत्या औषधामुळे किंवा उत्पादनामुळे हे झाले असावे असे तुम्हाला वाटते?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "तुम्ही कोणता दुष्परिणाम किंवा प्रतिक्रिया पाहिली ते मला सांगू शकाल का?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "हे अनुभवलेले तुम्ही आहात, की तुम्ही अजून एखाद्यासाठी सांगत आहात?"
    ),
    "What is the patient's full name?": "रुग्णाचे पूर्ण नाव काय आहे?",
    "May I have your first and last name?": "मी तुमचे पहिले व शेवटचे नाव जाणू शकतो का?",
    "How old is the patient, or what is their date of birth?": (
        "रुग्णाचे वय किती आहे, किंवा त्यांची जन्मतारीख काय आहे?"
    ),
    "How old are you, or what is your date of birth?": (
        "तुमचे वय किती आहे, किंवा तुमची जन्मतारीख काय आहे?"
    ),
    "And what is your name?": "आणि तुमचे नाव काय आहे?",
    "If we need a little more information later, is it alright if we contact you?": (
        "पुढे आम्हाला थोडी अधिक माहिती लागली, तर आम्ही तुमच्याशी संपर्क करू का?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "तुम्हाला फोनने संपर्क करावा वाटतो की ईमेलने?"
    ),
    "What phone number would you like us to use?": "आम्ही कोणता फोन नंबर वापरावा?",
    "What email address would you like us to use?": "आम्ही कोणता ईमेल पत्ता वापरावा?",
    "When did the side effect first start?": "दुष्परिणाम पहिल्यांदा कधी सुरू झाला?",
    "Would you consider this problem serious?": "तुम्हाला ही समस्या गंभीर वाटते का?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "रुग्ण आता कसा वाटत आहे - लक्षणे सुधारली आहेत, तशीच आहेत, की अधिक बिघडली आहेत?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "तुम्हाला आता कसे वाटत आहे - तुमची लक्षणे सुधारली आहेत, तशीच आहेत, की अधिक बिघडली आहेत?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "क्षमस्व, मला समजले नाही - ते बरोबर असल्यास होय म्हणा, आणि बदलवायचे असल्यास नाही म्हणा."
    ),
    "Is that correct?": "ते बरोबर आहे का?",
    "Thank you - I've recorded everything we need for now.": (
        "धन्यवाद - आत्ताच्या घडीला लागणारे सर्व नोंदवले आहे."
    ),
    "A member of our team will follow up if needed.": (
        "आवश्यक असल्यास आमच्या टीमेतील एक सदस्य तुमच्याशी संपर्क करेल."
    ),
    "Take care.": "तुमची काळजी घ्या.",
    "Thank you.": "धन्यवाद.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "संपवण्यापूर्वी, काय घडले याबद्दल अजून काही सांगायचे आहे का?"
    ),
    "Thank you for sharing that.": "ते सांगितल्याबद्दल धन्यवाद.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "ही वैद्यकीय आपत्कालीन परिस्थिती असल्यास, कृपया लगेच वैद्यकीय मदत घ्या "
        "किंवा तुमच्या स्थानिक आपत्कालीन सेवेला कॉल करा."
    ),
    "This appears potentially urgent.": "ही परिस्थिती संभाव्य तातडीची वाटते.",
    "A healthcare professional should be contacted immediately.": (
        "लगेच आरोग्य व्यावसायिकाशी संपर्क साधावा."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "मी वैयक्तिक डोस, उपचार शिफारसी किंवा औषध बदल सांगू शकत नाही."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "तुमच्या औषधांमध्ये कोणताही बदल करण्यासाठी कृपया योग्य "
        "आरोग्य व्यावसायिकाचा सल्ला घ्या."
    ),
    "I'm not sure how to interpret that.": "मला त्याचा अर्थ कळला नाही.",
    "Could you please clarify?": "कृपया तुम्ही स्पष्ट करू शकाल का?",
    "Sorry, I didn't catch the product name.": "क्षमस्व, मला उत्पादनाचे नाव समजले नाही.",
    "Which medicine were you using?": "तुम्ही कोणती औषध वापरत होता?",
    "Sorry, could you describe the side effect in a few words?": (
        "क्षमस्व, तुम्ही दुष्परिणाम थोड्या शब्दांत सांगू शकाल का?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "तपासणीसाठी - तुम्ही तुमचाच अनुभव सांगत आहात, की अजून एखाद्याचा?"
    ),
    "Could you say your first and last name again?": (
        "तुमचे पहिले व शेवटचे नाव पुन्हा सांगू शकाल का?"
    ),
    "How old are you, in years?": "तुमचे वय किती आहे, वर्षांत?",
    "Could you tell me your age in years, or your date of birth?": (
        "तुम्ही वर्षांत तुमचे वय किंवा जन्मतारीख सांगू शकाल का?"
    ),
    "Could you say the date of birth again?": "जन्मतारीख पुन्हा सांगू शकाल का?",
    "Would it be alright if we contact you later - yes or no?": (
        "आम्ही पुढे तुमच्याशी संपर्क करू का - होय की नाही?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "तुम्हाला कोणते सोपे आहे - फोन, ईमेल, की दोन्ही?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "फोन नंबर अंकानुसार पुन्हा सांगू शकाल का?"
    ),
    "Could you repeat the email address?": "ईमेल पत्ता पुन्हा सांगू शकाल का?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "तो जवळपास कधी सुरू झाला - जसे आज, काल, की काही दिवसांपूर्वी?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "तुम्हाला तो गंभीर वाटतो का - होय, नाही, की नक्की नाही?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "तुमची लक्षणे आता सुधारली आहेत, तशीच आहेत, की अधिक बिघडली आहेत?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "त्यासाठी तुम्ही वैद्यकीय मदत घेतली होती का - होय की नाही?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "रुग्ण आता कसा वाटत आहे - लक्षणे सुधारली आहेत, तशीच आहेत, की अधिक बिघडली आहेत?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "तुम्हाला आता कसे वाटत आहे - तुमची लक्षणे सुधारली आहेत, तशीच आहेत, की अधिक बिघडली आहेत?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "तुम्हाला फोनने, ईमेलने, की दोन्हीनी संपर्क करावा वाटतो?"
    ),
    "Thank you for sharing this information.": "ही माहिती शेअर केल्याबद्दल धन्यवाद.",
    "We've recorded what you've provided.": "तुम्ही दिलेली माहिती आम्ही नोंदवली आहे.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "तुम्ही पुढील संपर्कासाठी संमती दिलेली नाही, त्यामुळे या तपशिलांचा वापर "
        "करून आम्ही तुमच्याशी संपर्क करणार नाही."
    ),
    "Thank you for your patience.": "तुमच्या संयमाबद्दल धन्यवाद.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "आम्हाला काही तपशील निश्चित करता आले नाहीत, त्यामुळे तुमचा प्रकरणाचा "
        "आढावा आमच्या टीमेतील एक सदस्य घेईल."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "ते संपूर्ण पत्ता असेल तर होय म्हणा - अधिक अक्षरे बाकी असतील तर कृपया पत्ता पुन्हा सांगा."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "ते बरोबर नसेल, तर कृपया ते पुन्हा एक-एक अक्षर किंवा अंक म्हणून सांगा."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "आम्हाला नावाची खात्री पटली नाही."
    ),
    "We could not confirm the phone number.": (
        "आम्हाला फोन नंबरची खात्री पटली नाही."
    ),
    "We could not confirm the contact number.": (
        "आम्हाला संपर्क क्रमांकाची खात्री पटली नाही."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "तुम्हीं ते आता पुन्हा सांगू शकता, किंवा समीक्षेदरम्यान आमची टीम ते घेईल."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "हे MedVaani प्रोटोटाइप फक्त Lantus SoloStar साठी तयार केले आहे."
    ),
    "This prototype cannot continue this report.": "हे प्रोटोटाइप हा अहवाल पुढे चालू ठेवू शकत नाही.",
    "Thank you for taking the time to speak with us.": "आमच्याशी बोलण्यासाठी वेळ दिल्याबद्दल धन्यवाद.",
    "I'm here to help record the event for review.": (
        "आढाव्यासाठी घटना नोंदवण्यात मदत करण्यासाठी मी येथे आहे."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "मी तुम्हाला काही प्रश्न विचारीन, आणि आपण आढाव्यासाठी एक अहवाल तयार करू."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "मी आरोग्य व्यावसायिकाचा पर्याय नाही आणि वैद्यकीय सल्ला देऊ शकत नाही."
    ),
    "Please tell me what happened, in your own words.": "कृपया आपल्या शब्दांत सांगा की काय घडले.",
    "Take your time\u2014I'm listening.": "तुमच्या वेळेने सांगा\u2014मी ऐकत आहे.",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "तुमचा अहवाल बरोबर नोंदवण्यात मदतीसाठी, संबंधित औषधाचे नाव सांगू शकाल का?"
    ),
    "I'm sorry to hear that you've experienced this.": "तुम्ही हा अनुभव घेतला याबद्दल मला खेद आहे.",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "बरोबर औषध माझ्याकडे आहे याची खात्री करण्यासाठी, ते Lantus SoloStar आहे का?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "ही घटना अनुभवलेले तुम्ही आहात, की तुम्ही अजून एखाद्याच्या वतीने सांगत आहात "
        "असे मी विचारू शकतो का?"
    ),
    "Could you please tell me your full name?": "कृपया तुम्ही तुमचे पूर्ण नाव सांगू शकाल का?",
    "Could you please share your contact number?": "कृपया तुम्ही तुमचा संपर्क क्रमांक सांगू शकाल का?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "कृपया तुम्ही रुग्णाचे पहिले व शेवटचे नाव, सोबत जन्मतारीख सांगू शकाल का?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "कृपया तुम्ही तुमचे पहिले व शेवटचे नाव, सोबत जन्मतारीख सांगू शकाल का?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "कृपया तुम्ही रुग्णाची जन्मतारीख सांगू शकाल का?"
    ),
    "Could you please tell me your date of birth?": "कृपया तुम्ही तुमची जन्मतारीख सांगू शकाल का?",
    "Was it the name or the date of birth that was incorrect?": "नाव चुकीचे होते की जन्मतारीख?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "क्षमस्व, कृपया सांगा की नाव की जन्मतारीख यातील काय चुकीचे होते."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "तपासणीसाठी - तो अजून एखादे लक्षण होते, की तुम्ही सांगितलेल्या गोष्टीत दुरुस्ती होती?"
    ),
    "Sorry, please tell me a full date of birth.": "क्षमस्व, कृपया पूर्ण जन्मतारीख सांगा.",
    "Sorry, I couldn't catch that.": (
        "क्षमस्व, मला समजले नाही."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "मला महिना स्पष्ट ऐकू दिला नाही - कृपया जन्मतारीख दिवस, महिना आणि वर्ष असे सांगा."
    ),
    "Could you please repeat it?": (
        "कृपया तुम्ही पुन्हा सांगू शकाल का?"
    ),
    "I couldn't catch the email address.": (
        "मला ईमेल पत्ता समजले नाही."
    ),
    "Could you please share the patient's contact number?": (
        "कृपया तुम्ही रुग्णाचा संपर्क क्रमांक सांगू शकाल का?"
    ),
    "Could you please share the best contact number for you?": (
        "कृपया तुम्ही तुमच्यासाठी सर्वात योग्य संपर्क क्रमांक सांगू शकाल का?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "संपर्क क्रमांक अंकानुसार पुन्हा सांगू शकाल का?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "पुढे अधिक माहिती लागल्यास, संबंधित टीम या अहवालाबाबत रुग्णाशी "
        "संपर्क करू शकेल का?"
    ),
    "May I have the patient's email address, if available?": (
        "उपलब्ध असल्यास, रुग्णाचा ईमेल पत्ता मिळू शकेल का?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "कृपया तो ईमेल पत्ता सांगा जो तुम्ही नोंदवू इच्छिता."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "तुम्ही तुमचा ईमेल पत्ता एक-एक अक्षर किंवा अंक करून सांगू शकाल का?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "तुम्ही '@' साठी 'at the rate' आणि ठिपक्यासाठी 'dot' असे म्हणू शकता."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "तुम्ही म्हणाल की ही घटना गंभीर किंवा जीवघेणी होती?"
    ),
    "Please seek immediate medical attention now.": "कृपया आत्ताच तातडीने वैद्यकीय मदत घ्या.",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "तुम्ही आधीच पुढील संपर्कासाठी संमती दिली असल्यास, अधिक माहिती अधिक "
        "सोयीच्या वेळी घेण्याची आम्ही व्यवस्था करू."
    ),
    "Could you tell me when the event occurred?": "तुम्ही सांगू शकाल का की ही घटना कधी घडली?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "घटना नोंदवण्यात मदत होईल असे अजून काही तुम्ही सांगायचे आहे का?"
    ),
    "Of course.": "नक्कीच.",
    "Please tell me any additional details you'd like included.": (
        "कृपया तुम्हाला समाविष्ट करायचे असलेले अतिरिक्त तपशील सांगा."
    ),
    "Thank you for sharing this information with us.": "आमच्यासोबत ही माहिती शेअर केल्याबद्दल धन्यवाद.",
    "We've prepared the report for review.": "आम्ही आढाव्यासाठी अहवाल तयार केला आहे.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "तुम्ही पुढील संपर्कासाठी संमती दिली असल्यास, आवश्यक असल्यास संबंधित टीम "
        "तुमच्याशी संपर्क करू शकते."
    ),
    "Is that right?": "ते बरोबर आहे का?",
    "Have I got that right?": "मी बरोबर समजले आहे का?",
    "Have I recorded those correctly?": "मी ती बरोबर नोंदवली आहेत का?",
}

_MARATHI_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "पत्ता {0} असा वाचला गेला आहे.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "पत्ता अजूनही {0} च आहे.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "मी तुमचा ईमेल पत्ता {0} असा समजला.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "मी तुमचा ईमेल पत्ता {0} ऐकला.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "मी {0} ऐकले."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "तुमचे नाव {0} असे माझ्याकडे नोंदवले आहे.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "मी तुमचे तपशील असे नोंदवले आहेत: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "बरोबर नोंदवले आहे का याची खात्री करण्यासाठी मी पुन्हा वाचून दाखवतो: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "मला {0} अंकांचा नंबर मिळाला आहे, ज्याचा शेवटचा भाग {1} आहे.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "क्षमस्व, मला समजले नाही - कृपया बरोबर {0} सांगा.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "तुम्ही {0} बद्दल थोडे अधिक सांगू शकाल का?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "तुमची जन्मतारीख ({0}) आणि तुमचे वय ({1} वर्षे) जुळत नाहीत.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "मी कोणती गोष्ट नोंदवावी - तुमची जन्मतारीख की तुमचे वय?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0} साठी, कृपया औषधाच्या निर्मात्याशी किंवा तुमच्या स्थानिक "
        "प्रतिकूल घटना नोंदवण्याच्या माध्यमाशी संपर्क करा.",
    ),
)

# ---- Punjabi (pa-IN) ----------------------------------------------------------

_PUNJABI_SENTENCES: dict[str, str] = {
    "Thanks for that.": "ਇਸ ਲਈ ਧੰਨਵਾਦ।",
    "Got it, thank you.": "ਸਮਝ ਗਿਆ, ਧੰਨਵਾਦ।",
    "Okay, thank you.": "ਠੀਕ ਹੈ, ਧੰਨਵਾਦ।",
    "Thanks for the details.": "ਵੇਰਵੇ ਦੇਣ ਲਈ ਧੰਨਵਾਦ।",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "ਮੈਨੂੰ ਦੱਸਣ ਲਈ ਧੰਨਵਾਦ, ਅਤੇ ਮੈਨੂੰ ਦੁੱਖ ਹੈ ਕਿ ਤੁਸੀਂ ਇਹ ਸਭ ਝੱਲ ਰਹੇ ਹੋ।"
    ),
    "I'll take a few details so we can record it safely.": (
        "ਅਸੀਂ ਇਸਨੂੰ ਸੁਰੱਖਿਅਤ ਢੰਗ ਨਾਲ ਦਰਜ ਕਰਨ ਲਈ ਕੁਝ ਵੇਰਵੇ ਲੈਂਦੇ ਹਾਂ।"
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "ਮਹੱਤਵਪੂਰਨ: ਦੱਸੇ ਗਏ ਸਾਈਡ ਇਫੈਕਟਾਂ ਦੇ ਮੁਲਾਂਕਣ ਅਤੇ ਇਲਾਜ ਲਈ ਕਿਰਪਾ ਕਰਕੇ "
        "ਕਿਸੇ ਯੋਗ ਡਾਕਟਰ ਜਾਂ ਸਿਹਤ ਪੇਸ਼ੇਵਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "ਮੈਡੀਕਲ ਸਲਾਹ ਤੋਂ ਬਿਨਾਂ ਆਪਣੀ ਬਿਮਾਰੀ ਦਾ ਆਪ ਨਿਰਣਾ ਨਾ ਕਰੋ ਅਤੇ ਨਾ ਹੀ ਆਪਣੀ ਦਵਾਈ ਬਦਲੋ।"
    ),
    "Which medicine or product do you think may have caused this?": (
        "ਤੁਹਾਨੂੰ ਕਿਹੜੀ ਦਵਾਈ ਜਾਂ ਉਤਪਾਦ ਨੇ ਇਹ ਕੀਤਾ ਹੋ ਸਕਦਾ ਹੈ ਅਜਿਹਾ ਲੱਗਦਾ ਹੈ?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "ਤੁਸੀਂ ਕਿਹੜਾ ਸਾਈਡ ਇਫੈਕਟ ਜਾਂ ਪ੍ਰਤੀਕਿਰਿਆ ਵੇਖੀ, ਇਹ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "ਕੀ ਤੁਸੀਂ ਹੀ ਉਹ ਵਿਅਕਤੀ ਹੋ ਜਿਸ ਨਾਲ ਇਹ ਵਾਪਰਿਆ, ਜਾਂ ਤੁਸੀਂ ਕਿਸੇ ਹੋਰ ਦੀ ਤਰਫੋਂ ਦੱਸ ਰਹੇ ਹੋ?"
    ),
    "What is the patient's full name?": "ਮਰੀਜ਼ ਦਾ ਪੂਰਾ ਨਾਮ ਕੀ ਹੈ?",
    "May I have your first and last name?": "ਕੀ ਤੁਹਾਡਾ ਪਹਿਲਾ ਅਤੇ ਆਖਰੀ ਨਾਮ ਮਿਲ ਸਕਦਾ ਹੈ?",
    "How old is the patient, or what is their date of birth?": (
        "ਮਰੀਜ਼ ਦੀ ਉਮਰ ਕਿੰਨੀ ਹੈ, ਜਾਂ ਉਸਦੀ ਜਨਮ ਤਾਰੀਖ ਕੀ ਹੈ?"
    ),
    "How old are you, or what is your date of birth?": (
        "ਤੁਹਾਡੀ ਉਮਰ ਕਿੰਨੀ ਹੈ, ਜਾਂ ਤੁਹਾਡੀ ਜਨਮ ਤਾਰੀਖ ਕੀ ਹੈ?"
    ),
    "And what is your name?": "ਅਤੇ ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
    "If we need a little more information later, is it alright if we contact you?": (
        "ਜੇ ਸਾਨੂੰ ਬਾਅਦ ਵਿੱਚ ਥੋੜ੍ਹੀ ਜਿਹੀ ਹੋਰ ਜਾਣਕਾਰੀ ਚਾਹੀਦੀ ਹੈ, ਤਾਂ ਕੀ ਅਸੀਂ "
        "ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰ ਸਕਦੇ ਹਾਂ?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "ਫੋਨ ਰਾਹੀਂ ਸੰਪਰਕ ਠੀਕ ਹੈ ਜਾਂ ਈਮੇਲ ਰਾਹੀਂ?"
    ),
    "What phone number would you like us to use?": "ਅਸੀਂ ਕਿਹੜਾ ਫੋਨ ਨੰਬਰ ਵਰਤੀਏ?",
    "What email address would you like us to use?": "ਅਸੀਂ ਕਿਹੜਾ ਈਮੇਲ ਪਤਾ ਵਰਤੀਏ?",
    "When did the side effect first start?": "ਸਾਈਡ ਇਫੈਕਟ ਪਹਿਲੀ ਵਾਰ ਕਦੋਂ ਸ਼ੁਰੂ ਹੋਇਆ?",
    "Would you consider this problem serious?": "ਕੀ ਤੁਸੀਂ ਇਸ ਸਮੱਸਿਆ ਨੂੰ ਗੰਭੀਰ ਸਮਝਦੇ ਹੋ?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "ਮਰੀਜ਼ ਨੂੰ ਹੁਣ ਕਿਵੇਂ ਮਹਿਸੂਸ ਹੋ ਰਿਹਾ ਹੈ - ਲੱਛਣ ਸੁਧਰ ਗਏ ਹਨ, ਪਹਿਲਾਂ ਵਾਂਗ ਹਨ, ਜਾਂ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ ਹਨ?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "ਤੁਹਾਨੂੰ ਹੁਣ ਕਿਵੇਂ ਮਹਿਸੂਸ ਹੋ ਰਿਹਾ ਹੈ - ਤੁਹਾਡੇ ਲੱਛਣ ਸੁਧਰ ਗਏ ਹਨ, ਪਹਿਲਾਂ ਵਾਂਗ ਹਨ, ਜਾਂ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ ਹਨ?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ - ਜੇ ਇਹ ਠੀਕ ਹੈ ਤਾਂ ਹਾਂ ਕਹੋ, ਅਤੇ ਜੇ ਬਦਲਣਾ ਹੈ ਤਾਂ ਨਹੀਂ ਕਹੋ।"
    ),
    "Is that correct?": "ਕੀ ਇਹ ਠੀਕ ਹੈ?",
    "Thank you - I've recorded everything we need for now.": (
        "ਧੰਨਵਾਦ - ਹੁਣ ਲਈ ਲੋੜੀਂਦਾ ਸਭ ਕੁਝ ਮੈਂ ਦਰਜ ਕਰ ਲਿਆ ਹੈ।"
    ),
    "A member of our team will follow up if needed.": (
        "ਜੇ ਲੋੜ ਹੋਈ ਤਾਂ ਸਾਡੀ ਟੀਮ ਦਾ ਕੋਈ ਮੈਂਬਰ ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰੇਗਾ।"
    ),
    "Take care.": "ਆਪਣਾ ਧਿਆਨ ਰੱਖੋ।",
    "Thank you.": "ਧੰਨਵਾਦ।",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "ਮੁੱਕਣ ਤੋਂ ਪਹਿਲਾਂ, ਕੀ ਹੋਇਆ ਇਸ ਬਾਰੇ ਹੋਰ ਕੁਝ ਸਾਂਝਾ ਕਰਨਾ ਹੈ?"
    ),
    "Thank you for sharing that.": "ਇਹ ਦੱਸਣ ਲਈ ਧੰਨਵਾਦ।",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "ਜੇ ਇਹ ਕੋਈ ਮੈਡੀਕਲ ਐਮਰਜੈਂਸੀ ਹੈ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਤੁਰੰਤ ਮੈਡੀਕਲ ਮਦਦ ਲਵੋ "
        "ਜਾਂ ਆਪਣੀ ਸਥਾਨਕ ਐਮਰਜੈਂਸੀ ਸੇਵਾ ਨੂੰ ਕਾਲ ਕਰੋ।"
    ),
    "This appears potentially urgent.": "ਇਹ ਸੰਭਾਵੀ ਤੌਰ 'ਤੇ ਐਮਰਜੈਂਸੀ ਜਿਹਾ ਲੱਗਦਾ ਹੈ।",
    "A healthcare professional should be contacted immediately.": (
        "ਤੁਰੰਤ ਕਿਸੇ ਸਿਹਤ ਪੇਸ਼ੇਵਰ ਨਾਲ ਸੰਪਰਕ ਕੀਤਾ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ।"
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "ਮੈਂ ਨਿੱਜੀ ਖੁਰਾਕ, ਇਲਾਜ ਦੀ ਸਿਫ਼ਾਰਸ਼ ਜਾਂ ਦਵਾਈ ਵਿੱਚ ਬਦਲਾਅ ਨਹੀਂ ਦੱਸ ਸਕਦਾ।"
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "ਆਪਣੀ ਦਵਾਈ ਦੇ ਕ੍ਰਮ ਵਿੱਚ ਕਿਸੇ ਵੀ ਤਬਦੀਲੀ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਯੋਗ "
        "ਸਿਹਤ ਪੇਸ਼ੇਵਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
    ),
    "I'm not sure how to interpret that.": "ਮੈਨੂੰ ਇਸਦਾ ਮਤਲਬ ਸਮਝ ਨਹੀਂ ਆਇਆ।",
    "Could you please clarify?": "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਸਪਸ਼ਟ ਕਰ ਸਕਦੇ ਹੋ?",
    "Sorry, I didn't catch the product name.": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਉਤਪਾਦ ਦਾ ਨਾਮ ਸਮਝ ਨਹੀਂ ਆਇਆ।",
    "Which medicine were you using?": "ਤੁਸੀਂ ਕਿਹੜੀ ਦਵਾਈ ਵਰਤ ਰਹੇ ਸੀ?",
    "Sorry, could you describe the side effect in a few words?": (
        "ਮਾਫ਼ ਕਰਨਾ, ਕੀ ਤੁਸੀਂ ਸਾਈਡ ਇਫੈਕਟ ਕੁਝ ਸ਼ਬਦਾਂ ਵਿੱਚ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "ਪੱਕਾ ਕਰਨ ਲਈ - ਤੁਸੀਂ ਆਪਣਾ ਹੀ ਤਜਰਬਾ ਦੱਸ ਰਹੇ ਹੋ, ਜਾਂ ਕਿਸੇ ਹੋਰ ਦਾ?"
    ),
    "Could you say your first and last name again?": (
        "ਕੀ ਤੁਸੀਂ ਆਪਣਾ ਪਹਿਲਾ ਅਤੇ ਆਖਰੀ ਨਾਮ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "How old are you, in years?": "ਤੁਹਾਡੀ ਉਮਰ ਕਿੰਨੀ ਹੈ, ਸਾਲਾਂ ਵਿੱਚ?",
    "Could you tell me your age in years, or your date of birth?": (
        "ਕੀ ਤੁਸੀਂ ਸਾਲਾਂ ਵਿੱਚ ਆਪਣੀ ਉਮਰ ਜਾਂ ਜਨਮ ਤਾਰੀਖ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you say the date of birth again?": "ਕੀ ਤੁਸੀਂ ਜਨਮ ਤਾਰੀਖ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "Would it be alright if we contact you later - yes or no?": (
        "ਕੀ ਅਸੀਂ ਬਾਅਦ ਵਿੱਚ ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰੀਏ - ਹਾਂ ਜਾਂ ਨਹੀਂ?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "ਤੁਹਾਡੇ ਲਈ ਕੀ ਸੌਖਾ ਹੈ - ਫੋਨ, ਈਮੇਲ, ਜਾਂ ਦੋਵੇਂ?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "ਕੀ ਤੁਸੀਂ ਫੋਨ ਨੰਬਰ ਅੰਕ ਦਰ ਅੰਕ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you repeat the email address?": "ਕੀ ਤੁਸੀਂ ਈਮੇਲ ਪਤਾ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "ਇਹ ਤਕਰੀਬਨ ਕਦੋਂ ਸ਼ੁਰੂ ਹੋਇਆ - ਜਿਵੇਂ ਅੱਜ, ਕੱਲ੍ਹ, ਜਾਂ ਕੁਝ ਦਿਨ ਪਹਿਲਾਂ?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "ਕੀ ਤੁਸੀਂ ਇਸਨੂੰ ਗੰਭੀਰ ਸਮਝਦੇ ਹੋ - ਹਾਂ, ਨਹੀਂ, ਜਾਂ ਪੱਕਾ ਨਹੀਂ?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "ਤੁਹਾਡੇ ਲੱਛਣ ਹੁਣ ਸੁਧਰ ਗਏ ਹਨ, ਪਹਿਲਾਂ ਵਾਂਗ ਹਨ, ਜਾਂ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ ਹਨ?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "ਕੀ ਤੁਹਾਨੂੰ ਇਸ ਲਈ ਕੋਈ ਮੈਡੀਕਲ ਦੇਖਭਾਲ ਮਿਲੀ - ਹਾਂ ਜਾਂ ਨਹੀਂ?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "ਮਰੀਜ਼ ਨੂੰ ਹੁਣ ਕਿਵੇਂ ਮਹਿਸੂਸ ਹੋ ਰਿਹਾ ਹੈ - ਲੱਛਣ ਸੁਧਰ ਗਏ ਹਨ, ਪਹਿਲਾਂ ਵਾਂਗ ਹਨ, ਜਾਂ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ ਹਨ?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "ਤੁਹਾਨੂੰ ਹੁਣ ਕਿਵੇਂ ਮਹਿਸੂਸ ਹੋ ਰਿਹਾ ਹੈ - ਤੁਹਾਡੇ ਲੱਛਣ ਸੁਧਰ ਗਏ ਹਨ, ਪਹਿਲਾਂ ਵਾਂਗ ਹਨ, ਜਾਂ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ ਹਨ?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "ਤੁਹਾਨੂੰ ਫੋਨ ਰਾਹੀਂ, ਈਮੇਲ ਰਾਹੀਂ, ਜਾਂ ਦੋਵਾਂ ਰਾਹੀਂ ਸੰਪਰਕ ਠੀਕ ਲੱਗਦਾ ਹੈ?"
    ),
    "Thank you for sharing this information.": "ਇਹ ਜਾਣਕਾਰੀ ਸਾਂਝੀ ਕਰਨ ਲਈ ਧੰਨਵਾਦ।",
    "We've recorded what you've provided.": "ਅਸੀਂ ਤੁਹਾਡੇ ਵੱਲੋਂ ਦਿੱਤੀ ਜਾਣਕਾਰੀ ਦਰਜ ਕਰ ਲਈ ਹੈ।",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "ਕਿਉਂਕਿ ਤੁਸੀਂ ਬਾਅਦ ਦੇ ਸੰਪਰਕ ਲਈ ਸਹਿਮਤੀ ਨਹੀਂ ਦਿੱਤੀ, ਅਸੀਂ ਇਨ੍ਹਾਂ ਵੇਰਵਿਆਂ "
        "ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਨਹੀਂ ਕਰਾਂਗੇ।"
    ),
    "Thank you for your patience.": "ਤੁਹਾਡੇ ਸਬਰ ਲਈ ਧੰਨਵਾਦ।",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "ਅਸੀਂ ਕੁਝ ਵੇਰਵਿਆਂ ਦੀ ਪੁਸ਼ਟੀ ਨਹੀਂ ਕਰ ਸਕੇ, ਇਸ ਲਈ ਤੁਹਾਡੇ ਕੇਸ ਦੀ ਸਾਡੀ "
        "ਟੀਮ ਦਾ ਕੋਈ ਮੈਂਬਰ ਸਮੀਖਿਆ ਕਰੇਗਾ।"
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "ਜੇ ਇਹ ਪੂਰਾ ਪਤਾ ਹੈ, ਤਾਂ ਹਾਂ ਕਹੋ - ਜੇ ਹੋਰ ਅੱਖਰ ਬਾਕੀ ਹਨ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਪਤਾ ਦੁਬਾਰਾ ਦੱਸੋ।"
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "ਜੇ ਇਹ ਸਹੀ ਨਹੀਂ ਹੈ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ ਦੁਬਾਰਾ ਇੱਕ-ਇੱਕ ਅੱਖਰ ਜਾਂ ਅੰਕ ਬਣਾ ਕੇ ਦੱਸੋ।"
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "ਸਾਨੂੰ ਨਾਮ ਦੀ ਪੁਸ਼ਟੀ ਨਹੀਂ ਹੋ ਸਕੀ।"
    ),
    "We could not confirm the phone number.": (
        "ਸਾਨੂੰ ਫੋਨ ਨੰਬਰ ਦੀ ਪੁਸ਼ਟੀ ਨਹੀਂ ਹੋ ਸਕੀ।"
    ),
    "We could not confirm the contact number.": (
        "ਸਾਨੂੰ ਸੰਪਰਕ ਨੰਬਰ ਦੀ ਪੁਸ਼ਟੀ ਨਹੀਂ ਹੋ ਸਕੀ।"
    ),
    "You can say it again now, or our team will collect it during review.": (
        "ਤੁਸੀਂ ਇਸਨੂੰ ਹੁਣੇ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ, ਜਾਂ ਸਮੀਖਿਆ ਦੌਰਾਨ ਸਾਡੀ ਟੀਮ ਇਸਨੂੰ ਲੈ ਲਵੇਗੀ।"
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "ਇਹ MedVaani ਪ੍ਰੋਟੋਟਾਈਪ ਸਿਰਫ਼ Lantus SoloStar ਲਈ ਤਿਆਰ ਕੀਤਾ ਗਿਆ ਹੈ।"
    ),
    "This prototype cannot continue this report.": "ਇਹ ਪ੍ਰੋਟੋਟਾਈਪ ਇਹ ਰਿਪੋਰਟ ਜਾਰੀ ਨਹੀਂ ਰੱਖ ਸਕਦਾ।",
    "Thank you for taking the time to speak with us.": "ਸਾਡੇ ਨਾਲ ਗੱਲ ਕਰਨ ਲਈ ਸਮਾਂ ਦੇਣ ਲਈ ਧੰਨਵਾਦ।",
    "I'm here to help record the event for review.": (
        "ਸਮੀਖਿਆ ਲਈ ਘਟਨਾ ਦਰਜ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰਨ ਲਈ ਮੈਂ ਇੱਥੇ ਹਾਂ।"
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "ਮੈਂ ਤੁਹਾਨੂੰ ਕੁਝ ਸਵਾਲ ਪੁੱਛਾਂਗਾ, ਅਤੇ ਅਸੀਂ ਸਮੀਖਿਆ ਲਈ ਇੱਕ ਰਿਪੋਰਟ ਤਿਆਰ ਕਰਾਂਗੇ।"
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "ਮੈਂ ਕਿਸੇ ਸਿਹਤ ਪੇਸ਼ੇਵਰ ਦਾ ਬਦਲ ਨਹੀਂ ਹਾਂ, ਅਤੇ ਮੈਂ ਮੈਡੀਕਲ ਸਲਾਹ ਨਹੀਂ ਦੇ ਸਕਦਾ।"
    ),
    "Please tell me what happened, in your own words.": "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਸ਼ਬਦਾਂ ਵਿੱਚ ਦੱਸੋ ਕਿ ਕੀ ਹੋਇਆ।",
    "Take your time\u2014I'm listening.": "ਆਪਣਾ ਸਮਾਂ ਲਵੋ\u2014ਮੈਂ ਸੁਣ ਰਿਹਾ ਹਾਂ।",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "ਤੁਹਾਡੀ ਰਿਪੋਰਟ ਸਹੀ ਢੰਗ ਨਾਲ ਦਰਜ ਕਰਨ ਵਿੱਚ ਮਦਦ ਲਈ, ਕੀ ਤੁਸੀਂ ਦੱਸ ਸਕਦੇ "
        "ਹੋ ਕਿ ਸ਼ਾਮਲ ਦਵਾਈ ਦਾ ਨਾਮ ਕੀ ਹੈ?"
    ),
    "I'm sorry to hear that you've experienced this.": "ਤੁਹਾਨੂੰ ਇਹ ਵਾਪਰਿਆ ਜਾਣ ਕੇ ਮੈਨੂੰ ਦੁੱਖ ਹੈ।",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "ਸਹੀ ਦਵਾਈ ਯਕੀਨੀ ਬਣਾਉਣ ਲਈ - ਕੀ ਇਹ Lantus SoloStar ਹੈ?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "ਕੀ ਮੈਂ ਪੁੱਛ ਸਕਦਾ ਹਾਂ ਕਿ ਕੀ ਤੁਸੀਂ ਹੀ ਉਹ ਵਿਅਕਤੀ ਹੋ ਜਿਸ ਨਾਲ ਇਹ ਘਟਨਾ "
        "ਵਾਪਰੀ, ਜਾਂ ਤੁਸੀਂ ਕਿਸੇ ਹੋਰ ਦੀ ਤਰਫੋਂ ਦੱਸ ਰਹੇ ਹੋ?"
    ),
    "Could you please tell me your full name?": "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਪੂਰਾ ਨਾਮ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "Could you please share your contact number?": "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਸੰਪਰਕ ਨੰਬਰ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਮਰੀਜ਼ ਦਾ ਪਹਿਲਾ ਅਤੇ ਆਖਰੀ ਨਾਮ, ਨਾਲ ਹੀ ਜਨਮ ਤਾਰੀਖ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਪਹਿਲਾ ਅਤੇ ਆਖਰੀ ਨਾਮ, ਨਾਲ ਹੀ ਆਪਣੀ ਜਨਮ ਤਾਰੀਖ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਮਰੀਜ਼ ਦੀ ਜਨਮ ਤਾਰੀਖ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you please tell me your date of birth?": "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਜਨਮ ਤਾਰੀਖ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "Was it the name or the date of birth that was incorrect?": "ਕੀ ਨਾਮ ਗਲਤ ਸੀ ਜਾਂ ਜਨਮ ਤਾਰੀਖ?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "ਮਾਫ਼ ਕਰਨਾ, ਕਿਰਪਾ ਕਰਕੇ ਦੱਸੋ ਕਿ ਨਾਮ ਜਾਂ ਜਨਮ ਤਾਰੀਖ ਵਿੱਚੋਂ ਕੀ ਗਲਤ ਸੀ।"
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "ਪੱਕਾ ਕਰਨ ਲਈ - ਕੀ ਇਹ ਕੋਈ ਹੋਰ ਲੱਛਣ ਸੀ, ਜਾਂ ਤੁਹਾਡੇ ਦੱਸੇ ਗੱਲ ਵਿੱਚ ਸੋਧ ਸੀ?"
    ),
    "Sorry, please tell me a full date of birth.": "ਮਾਫ਼ ਕਰਨਾ, ਕਿਰਪਾ ਕਰਕੇ ਪੂਰੀ ਜਨਮ ਤਾਰੀਖ ਦੱਸੋ।",
    "Sorry, I couldn't catch that.": (
        "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ।"
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "ਮੈਨੂੰ ਮਹੀਨਾ ਸਾਫ਼ ਨਹੀਂ ਸੁਣਿਆ - ਕਿਰਪਾ ਕਰਕੇ ਜਨਮ ਮਿਤੀ ਦਿਨ, ਮਹੀਨਾ ਅਤੇ ਸਾਲ ਵਜੋਂ ਦੱਸੋ।"
    ),
    "Could you please repeat it?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "I couldn't catch the email address.": (
        "ਮੈਨੂੰ ਈਮੇਲ ਪਤਾ ਸਮਝ ਨਹੀਂ ਆਇਆ।"
    ),
    "Could you please share the patient's contact number?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਮਰੀਜ਼ ਦਾ ਸੰਪਰਕ ਨੰਬਰ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you please share the best contact number for you?": (
        "ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਸੰਪਰਕ ਨੰਬਰ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "ਕੀ ਤੁਸੀਂ ਸੰਪਰਕ ਨੰਬਰ ਅੰਕ ਦਰ ਅੰਕ ਦੁਬਾਰਾ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "ਜੇ ਹੋਰ ਜਾਣਕਾਰੀ ਦੀ ਲੋੜ ਹੋਈ, ਤਾਂ ਕੀ ਸਬੰਧਤ ਟੀਮ ਇਸ ਰਿਪੋਰਟ ਬਾਰੇ "
        "ਮਰੀਜ਼ ਦਾ ਸੰਪਰਕ ਕਰ ਸਕਦੀ ਹੈ?"
    ),
    "May I have the patient's email address, if available?": (
        "ਜੇ ਉਪਲਬਧ ਹੈ, ਤਾਂ ਕੀ ਮਰੀਜ਼ ਦਾ ਈਮੇਲ ਪਤਾ ਮਿਲ ਸਕਦਾ ਹੈ?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "ਕਿਰਪਾ ਕਰਕੇ ਉਹ ਈਮੇਲ ਪਤਾ ਦੱਸੋ ਜਿਸਨੂੰ ਤੁਸੀਂ ਦਰਜ ਕਰਵਾਉਣਾ ਚਾਹੁੰਦੇ ਹੋ।"
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "ਕੀ ਤੁਸੀਂ ਆਪਣਾ ਈਮੇਲ ਪਤਾ ਇੱਕ-ਇੱਕ ਅੱਖਰ ਜਾਂ ਅੰਕ ਕਰਕੇ ਦੱਸ ਸਕਦੇ ਹੋ?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "ਤੁਸੀਂ '@' ਲਈ 'at the rate' ਅਤੇ ਬਿੰਦੀ ਲਈ 'dot' ਕਹਿ ਸਕਦੇ ਹੋ।"
    ),
    "Would you say that the event was severe or life-threatening?": (
        "ਕੀ ਤੁਸੀਂ ਕਹੋਗੇ ਕਿ ਇਹ ਘਟਨਾ ਗੰਭੀਰ ਜਾਂ ਜਾਨਲੇਵਾ ਸੀ?"
    ),
    "Please seek immediate medical attention now.": "ਕਿਰਪਾ ਕਰਕੇ ਹੁਣੇ ਤੁਰੰਤ ਮੈਡੀਕਲ ਮਦਦ ਲਵੋ।",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "ਜੇ ਤੁਸੀਂ ਪਹਿਲਾਂ ਹੀ ਬਾਅਦ ਦੇ ਸੰਪਰਕ ਲਈ ਸਹਿਮਤੀ ਦਿੱਤੀ ਹੈ, ਤਾਂ ਅਸੀਂ ਵਾਧੂ "
        "ਜਾਣਕਾਰੀ ਵਧੇਰੇ ਸੁਵਿਧਾਜਨਕ ਸਮੇਂ 'ਤੇ ਇਕੱਠੀ ਕਰਨ ਦਾ ਪ੍ਰਬੰਧ ਕਰਾਂਗੇ।"
    ),
    "Could you tell me when the event occurred?": "ਕੀ ਤੁਸੀਂ ਦੱਸ ਸਕਦੇ ਹੋ ਕਿ ਇਹ ਘਟਨਾ ਕਦੋਂ ਵਾਪਰੀ?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "ਕੀ ਅਜਿਹੀ ਕੋਈ ਹੋਰ ਗੱਲ ਹੈ ਜੋ ਘਟਨਾ ਦਰਜ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕੇ ਅਤੇ "
        "ਤੁਸੀਂ ਸਾਂਝੀ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ?"
    ),
    "Of course.": "ਜ਼ਰੂਰ।",
    "Please tell me any additional details you'd like included.": (
        "ਕਿਰਪਾ ਕਰਕੇ ਉਹ ਸਾਰੇ ਵਾਧੂ ਵੇਰਵੇ ਦੱਸੋ ਜਿਨ੍ਹਾਂ ਨੂੰ ਤੁਸੀਂ ਸ਼ਾਮਲ ਕਰਵਾਉਣਾ ਚਾਹੁੰਦੇ ਹੋ।"
    ),
    "Thank you for sharing this information with us.": "ਸਾਡੇ ਨਾਲ ਇਹ ਜਾਣਕਾਰੀ ਸਾਂਝੀ ਕਰਨ ਲਈ ਧੰਨਵਾਦ।",
    "We've prepared the report for review.": "ਅਸੀਂ ਸਮੀਖਿਆ ਲਈ ਰਿਪੋਰਟ ਤਿਆਰ ਕਰ ਲਈ ਹੈ।",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "ਜੇ ਤੁਸੀਂ ਬਾਅਦ ਦੇ ਸੰਪਰਕ ਲਈ ਸਹਿਮਤੀ ਦਿੱਤੀ ਹੈ, ਤਾਂ ਲੋੜ ਪੈਣ 'ਤੇ ਸਬੰਧਤ "
        "ਟੀਮ ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰ ਸਕਦੀ ਹੈ।"
    ),
    "Is that right?": "ਕੀ ਇਹ ਠੀਕ ਹੈ?",
    "Have I got that right?": "ਕੀ ਮੈਂ ਸਹੀ ਸਮਝਿਆ?",
    "Have I recorded those correctly?": "ਕੀ ਮੈਂ ਉਨ੍ਹਾਂ ਨੂੰ ਸਹੀ ਢੰਗ ਨਾਲ ਦਰਜ ਕੀਤਾ ਹੈ?",
}

_PUNJABI_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "ਪਤਾ {0} ਵਜੋਂ ਪੜ੍ਹਿਆ ਗਿਆ ਹੈ।",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "ਪਤਾ ਹਾਲੇ ਵੀ {0} ਹੀ ਹੈ।",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "ਮੈਂ ਤੁਹਾਡਾ ਈਮੇਲ ਪਤਾ {0} ਸਮਝਿਆ।",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "ਮੈਂ ਤੁਹਾਡਾ ਈਮੇਲ ਪਤਾ {0} ਸੁਣਿਆ।",
    ),
    (re.compile(r"^I heard (.+)\.$"), "ਮੈਂ {0} ਸੁਣਿਆ।"),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "ਮੇਰੇ ਕੋਲ ਤੁਹਾਡਾ ਨਾਮ {0} ਦਰਜ ਹੈ।",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "ਮੈਂ ਤੁਹਾਡੇ ਵੇਰਵੇ ਇਸ ਤਰ੍ਹਾਂ ਦਰਜ ਕੀਤੇ ਹਨ: {0}।",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "ਸਹੀ ਦਰਜ ਕੀਤਾ ਹੈ ਜਾਂ ਨਹੀਂ, ਇਹ ਯਕੀਨੀ ਬਣਾਉਣ ਲਈ ਮੈਂ ਦੁਬਾਰਾ ਪੜ੍ਹ ਕੇ ਸੁਣਾਉਂਦਾ ਹਾਂ: {0}।",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "ਮੈਨੂੰ {0} ਅੰਕਾਂ ਵਾਲਾ ਨੰਬਰ ਮਿਲਿਆ ਹੈ, ਜਿਸ ਦਾ ਆਖਰੀ ਹਿੱਸਾ {1} ਹੈ।",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ - ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ {0} ਦੱਸੋ।",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "ਕੀ ਤੁਸੀਂ {0} ਬਾਰੇ ਥੋੜ੍ਹਾ ਹੋਰ ਦੱਸ ਸਕਦੇ ਹੋ?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "ਤੁਹਾਡੀ ਜਨਮ ਤਾਰੀਖ ({0}) ਅਤੇ ਤੁਹਾਡੀ ਉਮਰ ({1} ਸਾਲ) ਮੇਲ ਨਹੀਂ ਖਾਂਦੇ।",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "ਮੈਂ ਕੀ ਦਰਜ ਕਰਾਂ - ਤੁਹਾਡੀ ਜਨਮ ਤਾਰੀਖ ਜਾਂ ਤੁਹਾਡੀ ਉਮਰ?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0} ਲਈ, ਕਿਰਪਾ ਕਰਕੇ ਦਵਾਈ ਦੇ ਨਿਰਮਾਤਾ ਜਾਂ ਆਪਣੀ ਸਥਾਨਿਕ "
        "ਪ੍ਰਤਿਕੂਲ ਘਟਨਾ ਰਿਪੋਰਟਿੰਗ ਚੈਨਲ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
    ),
)

# ---- Tamil (ta-IN) ------------------------------------------------------------

_TAMIL_SENTENCES: dict[str, str] = {
    "Thanks for that.": "இதற்கு நன்றி.",
    "Got it, thank you.": "புரிந்தது, நன்றி.",
    "Okay, thank you.": "சரி, நன்றி.",
    "Thanks for the details.": "விவரங்களைச் சொன்னதற்கு நன்றி.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "எனக்குச் சொன்னதற்கு நன்றி; இதை நீங்கள் சந்திக்கிறீர்கள் என்பது "
        "எனக்கு வருத்தமளிக்கிறது."
    ),
    "I'll take a few details so we can record it safely.": (
        "இதை பாதுகாப்பாக பதிவு செய்ய நான் சில விவரங்களை எடுத்துக் கொள்கிறேன்."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "முக்கியம்: தெரிவிக்கப்பட்ட பக்க விளைவுகளை மதிப்பிடவும் சிகிச்சையளிக்கவும் "
        "தயவுசெய்து தகுதியான ஒரு மருத்துவரை அணுகவும்."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "மருத்துவ ஆலோசனை இல்லாமல் நீங்களே நோய் கண்டறிய வேண்டாம், "
        "உங்கள் மருந்தை மாற்ற வேண்டாம்."
    ),
    "Which medicine or product do you think may have caused this?": (
        "எந்த மருந்து அல்லது பொருளால் இது நேர்ந்திருக்கலாம் என்று நினைக்கிறீர்கள்?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "நீங்கள் கவனித்த பக்க விளைவு அல்லது எதிர்வினை என்ன என்று சொல்ல முடியுமா?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "இதை அனுபவித்தவர் நீங்களா, அல்லது வேறு ஒருவருக்காகத் தெரிவிக்கிறீர்களா?"
    ),
    "What is the patient's full name?": "நோயாளியின் முழு பெயர் என்ன?",
    "May I have your first and last name?": "உங்கள் முதல் மற்றும் கடைசி பெயரைத் தெரிந்து கொள்ளலாமா?",
    "How old is the patient, or what is their date of birth?": (
        "நோயாளிக்கு வயது என்ன, அல்லது அவர்களின் பிறந்த தேதி என்ன?"
    ),
    "How old are you, or what is your date of birth?": (
        "உங்களுக்கு வயது என்ன, அல்லது உங்கள் பிறந்த தேதி என்ன?"
    ),
    "And what is your name?": "உங்கள் பெயர் என்ன?",
    "If we need a little more information later, is it alright if we contact you?": (
        "பின்னர் இன்னும் கொஞ்சம் தகவல் தேவைப்பட்டால், உங்களைத் தொடர்பு கொள்ளலாமா?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "தொலைபேசி மூலமா, மின்னஞ்சல் மூலமா தொடர்பு கொள்ள விரும்புகிறீர்கள்?"
    ),
    "What phone number would you like us to use?": (
        "நாங்கள் எந்தத் தொலைபேசி எண்ணைப் பயன்படுத்த வேண்டும்?"
    ),
    "What email address would you like us to use?": (
        "நாங்கள் எந்த மின்னஞ்சல் முகவரியைப் பயன்படுத்த வேண்டும்?"
    ),
    "When did the side effect first start?": "பக்க விளைவு முதன்முதலில் எப்போது தொடங்கியது?",
    "Would you consider this problem serious?": (
        "இந்தப் பிரச்சினையை நீங்கள் கடுமையானதாகக் கருவீர்களா?"
    ),
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "நோயாளி இப்போது எப்படி உள்ளார் - அறிகுறிகள் முன்னேறியுள்ளனவா, "
        "அப்படியே உள்ளனவா, அல்லது மோசமடைந்துள்ளனவா?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "நீங்கள் இப்போது எப்படி இருக்கிறீர்கள் - உங்கள் அறிகுறிகள் முன்னேறியுள்ளனவா, "
        "அப்படியே உள்ளனவா, அல்லது மோசமடைந்துள்ளனவா?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "மன்னிக்கவும், எனக்குப் புரியவில்லை - அது சரியாக இருந்தால் ஆம் என்றும், "
        "மாற்ற வேண்டும் என்றால் இல்லை என்றும் சொல்லுங்கள்."
    ),
    "Is that correct?": "அது சரியா?",
    "Thank you - I've recorded everything we need for now.": (
        "நன்றி - இப்போதுக்குத் தேவையான அனைத்தையும் பதிவு செய்துவிட்டேன்."
    ),
    "A member of our team will follow up if needed.": (
        "தேவைப்பட்டால் எங்கள் குழுவின் ஒரு உறுப்பினர் உங்களைத் தொடர்பு கொள்வார்."
    ),
    "Take care.": "கவனமாக இருங்கள்.",
    "Thank you.": "நன்றி.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "முடிப்பதறகு முன், என்ன நடந்தது என்று வேறு ஏதேனும் பகிர விரும்புகிறீர்களா?"
    ),
    "Thank you for sharing that.": "அதைப் பகிர்ந்ததற்கு நன்றி.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "இது ஒரு மருத்துவ அவசரநிலை என்றால், தயவுசெய்து உடனடியாக மருத்துவ "
        "உதவியை நாடுங்கள் அல்லது உங்கள் உள்ளூர் அவசர சேவையை அழையுங்கள்."
    ),
    "This appears potentially urgent.": "இது சாத்தியமான அவசரநிலையாகத் தெரிகிறது.",
    "A healthcare professional should be contacted immediately.": (
        "உடனடியாக ஒரு சுகாதார நிபுணரைத் தொடர்பு கொள்ள வேண்டும்."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "தனிப்பயன் அளவு, சிகிச்சை பரிந்துரைகள் அல்லது மருந்து மாற்றங்களை "
        "நான் வழங்க முடியாது."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "உங்கள் மருந்து முறையில் ஏதேனும் மாற்றத்திற்கு தயவுசெய்து தகுதியான "
        "சுகாதார நிபுணரை அணுகவும்."
    ),
    "I'm not sure how to interpret that.": "அதை எப்படிப் புரிந்து கொள்வது என்று எனக்குத் தெரியவில்லை.",
    "Could you please clarify?": "தயவுசெய்து விளக்கமளிக்க முடியுமா?",
    "Sorry, I didn't catch the product name.": "மன்னிக்கவும், பொருளின் பெயர் எனக்குப் புரியவில்லை.",
    "Which medicine were you using?": "நீங்கள் எந்த மருந்தைப் பயன்படுத்தினீர்கள்?",
    "Sorry, could you describe the side effect in a few words?": (
        "மன்னிக்கவும், பக்க விளைவைச் சில வார்த்தைகளில் விவரிக்க முடியுமா?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "உறுதிப்படுத்த - நீங்கள் உங்கள் சொந்த அனுபவத்தையா தெரிவிக்கிறீர்கள், "
        "அல்லது வேறொருவருடையதா?"
    ),
    "Could you say your first and last name again?": (
        "உங்கள் முதல் மற்றும் கடைசி பெயரை மீண்டும் சொல்ல முடியுமா?"
    ),
    "How old are you, in years?": "உங்களுக்கு எவ்வளவு வயது, ஆண்டுகளில்?",
    "Could you tell me your age in years, or your date of birth?": (
        "ஆண்டுகளில் உங்கள் வயது அல்லது உங்கள் பிறந்த தேதியைச் சொல்ல முடியுமா?"
    ),
    "Could you say the date of birth again?": "பிறந்த தேதியை மீண்டும் சொல்ல முடியுமா?",
    "Would it be alright if we contact you later - yes or no?": (
        "நாங்கள் பின்னர் உங்களைத் தொடர்பு கொண்டால் சரியாக இருக்குமா - ஆம் அல்லது இல்லை?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "உங்களுக்கு எது எளிது - தொலைபேசி, மின்னஞ்சல், அல்லது இரண்டும்?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "தொலைபேசி எண்ணை இலக்கம் இலக்கமாக மீண்டும் சொல்ல முடியுமா?"
    ),
    "Could you repeat the email address?": "மின்னஞ்சல் முகவரியை மீண்டும் சொல்ல முடியுமா?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "இது தோராயமாக எப்போது தொடங்கியது - உதாரணமாக இன்று, நேற்று, "
        "அல்லது சில நாட்களுக்கு முன்பா?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "இதை நீங்கள் கடுமையானதாகக் கருதுகிறீர்களா - ஆம், இல்லை, அல்லது "
        "உறுதியாகத் தெரியவில்லை?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "உங்கள் அறிகுறிகள் இப்போது முன்னேறியுள்ளனவா, அப்படியே உள்ளனவா, "
        "அல்லது மோசமடைந்துள்ளனவா?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "இதற்காக நீங்கள் எந்த மருத்துவ கவனிப்பையும் பெற்றீர்களா - ஆம் அல்லது இல்லை?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "நோயாளி இப்போது எப்படி உள்ளார் - அறிகுறிகள் முன்னேறியுள்ளனவா, "
        "அப்படியே உள்ளனவா, அல்லது மோசமடைந்துள்ளனவா?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "நீங்கள் இப்போது எப்படி இருக்கிறீர்கள் - உங்கள் அறிகுறிகள் முன்னேறியுள்ளனவா, "
        "அப்படியே உள்ளனவா, அல்லது மோசமடைந்துள்ளனவா?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "தொலைபேசி மூலமா, மின்னஞ்சல் மூலமா, அல்லது இரண்டும் மூலமா "
        "தொடர்பு கொள்ள விரும்புகிறீர்கள்?"
    ),
    "Thank you for sharing this information.": "இந்தத் தகவலைப் பகிர்ந்ததற்கு நன்றி.",
    "We've recorded what you've provided.": "நீங்கள் வழங்கிய தகவலை நாங்கள் பதிவு செய்துள்ளோம்.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "நீங்கள் பின்தொடர்புக்கு சம்மதிக்காததால், இந்த விவரங்களைப் பயன்படுத்தி "
        "நாங்கள் உங்களைத் தொடர்பு கொள்ள மாட்டோம்."
    ),
    "Thank you for your patience.": "உங்கள் பொறுமைக்கு நன்றி.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "சில விவரங்களை உறுதிப்படுத்த எங்களால் முடியவில்லை; எனவே உங்கள் "
        "வழக்கை எங்கள் குழுவின் ஒரு உறுப்பினர் மறுஆய்வு செய்வார்."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "அது முழு முகவரியாக இருந்தால் ஆம் எனச் சொல்லுங்கள் - மேலும் எழுத்துகள் இருந்தால் தயவுசெய்து முகவரியை மீண்டும் சொல்லுங்கள்."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "அது சரியில்லையெனில், தயவுசெய்து அதை மீண்டும் ஒவ்வொரு எழுத்தாகவோ எண்ணாகவோ சொல்லுங்கள்."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "எங்களால் பெயரை உறுதிப்படுத்த முடியவில்லை."
    ),
    "We could not confirm the phone number.": (
        "எங்களால் போன் எண்ணை உறுதிப்படுத்த முடியவில்லை."
    ),
    "We could not confirm the contact number.": (
        "எங்களால் தொடர்பு எண்ணை உறுதிப்படுத்த முடியவில்லை."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "நீங்கள் இப்போது அதை மீண்டும் சொல்லலாம், அல்லது மறுஆய்வின் போது எங்கள் குழு அதை பெறும்."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "இந்த MedVaani முன்மாதிரி Lantus SoloStar-க்கு மட்டுமே அமைக்கப்பட்டுள்ளது."
    ),
    "This prototype cannot continue this report.": (
        "இந்த முன்மாதிரிக்கு இந்த அறிக்கையைத் தொடர முடியாது."
    ),
    "Thank you for taking the time to speak with us.": "எங்களுடன் பேச நேரம் ஒதுக்கியதற்கு நன்றி.",
    "I'm here to help record the event for review.": (
        "மறுஆய்வுக்காக நிகழ்வைப் பதிவு செய்ய உதவ நான் இங்கே இருக்கிறேன்."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "நான் உங்களிடம் சில கேள்விகளைக் கேப்பேன்; மறுஆய்வுக்காக ஒரு "
        "அறிக்கையைத் தயார் செய்வோம்."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "நான் ஒரு சுகாதார நிபுணருக்கு மாற்றாக இல்லை; மருத்துவ ஆலோசனை "
        "வழங்க முடியாது."
    ),
    "Please tell me what happened, in your own words.": (
        "என்ன நடந்தது என்று உங்கள் சொந்த வார்த்தைகளில் சொல்லுங்கள்."
    ),
    "Take your time\u2014I'm listening.": (
        "நேரம் எடுத்துக் கொள்ளுங்கள்\u2014நான் கேட்டுக் கொண்டிருக்கிறேன்."
    ),
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "உங்கள் அறிக்கையை சரியாகப் பதிவு செய்ய உதவ, தொடர்புடைய மருந்தின் "
        "பெயரைச் சொல்ல முடியுமா?"
    ),
    "I'm sorry to hear that you've experienced this.": (
        "நீங்கள் இதை அனுபவித்ததைக் கேட்க வருத்தமாக இருக்கிறது."
    ),
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "சரியான மருந்து என்னிடம் இருப்பதை உறுதிப்படுத்த - அது Lantus SoloStarதானா?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "இந்த நிகழ்வை அனுபவித்தவர் நீங்களா, அல்லது வேறு ஒருவருக்காகத் "
        "தெரிவிக்கிறீர்களா என்று கேட்கலாமா?"
    ),
    "Could you please tell me your full name?": "தயவுசெய்து உங்கள் முழு பெயரைச் சொல்ல முடியுமா?",
    "Could you please share your contact number?": "தயவுசெய்து உங்கள் தொடர்பு எண்ணைச் சொல்ல முடியுமா?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "தயவுசெய்து நோயாளியின் முதல் மற்றும் கடைசி பெயர், பிறந்த தேதியுடன் "
        "சொல்ல முடியுமா?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "தயவுசெய்து உங்கள் முதல் மற்றும் கடைசி பெயர், பிறந்த தேதியுடன் "
        "சொல்ல முடியுமா?"
    ),
    "Could you please tell me the patient's date of birth?": (
        "தயவுசெய்து நோயாளியின் பிறந்த தேதியைச் சொல்ல முடியுமா?"
    ),
    "Could you please tell me your date of birth?": "தயவுசெய்து உங்கள் பிறந்த தேதியைச் சொல்ல முடியுமா?",
    "Was it the name or the date of birth that was incorrect?": "பெயரா தவறாக இருந்தது, பிறந்த தேதியா?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "மன்னிக்கவும், பெயர் அல்லது பிறந்த தேதியில் எது தவறாக இருந்தது "
        "என்று சொல்லுங்கள்."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "உறுதிப்படுத்த - அது வேறொரு அறிகுறியாக இருந்ததா, அல்லது நீங்கள் "
        "விவரித்ததில் திருத்தமா?"
    ),
    "Sorry, please tell me a full date of birth.": "மன்னிக்கவும், தயவுசெய்து முழு பிறந்த தேதியைச் சொல்லுங்கள்.",
    "Sorry, I couldn't catch that.": (
        "மன்னிக்கவும், எனக்குப் புரியவில்லை."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "மாதம் தெளிவாகக் கேட்கவில்லை - தயவுசெய்து பிறந்த தேதியை நாள், மாதம், ஆண்டு எனச் சொல்லுங்கள்."
    ),
    "Could you please repeat it?": (
        "தயவுசெய்து மீண்டும் சொல்ல முடியுமா?"
    ),
    "I couldn't catch the email address.": (
        "மின்னஞ்சல் முகவரி எனக்குப் புரியவில்லை."
    ),
    "Could you please share the patient's contact number?": (
        "தயவுசெய்து நோயாளியின் தொடர்பு எண்ணைச் சொல்ல முடியுமா?"
    ),
    "Could you please share the best contact number for you?": (
        "தயவுசெய்து உங்களுக்கு ஏற்ற தொடர்பு எண்ணைச் சொல்ல முடியுமா?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "தொடர்பு எண்ணை இலக்கம் இலக்கமாக மீண்டும் சொல்ல முடியுமா?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "மேலும் தகவல் தேவைப்பட்டால், தொடர்புடைய குழு இந்த அறிக்கை குறித்து "
        "நோயாளியைத் தொடர்பு கொள்ளலாமா?"
    ),
    "May I have the patient's email address, if available?": (
        "கிடைத்தால், நோயாளியின் மின்னஞ்சல் முகவரியைத் தெரிந்து கொள்ளலாமா?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "நீங்கள் பதிவு செய்ய விரும்பும் மின்னஞ்சல் முகவரியைத் தயவுசெய்து சொல்லுங்கள்."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "உங்கள் மின்னஞ்சல் முகவரியை ஒவ்வொரு எழுத்தாகவோ இலக்கமாகவோ "
        "சொல்ல முடியுமா?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "'@'க்கு 'at the rate' என்றும், புள்ளிக்கு 'dot' என்றும் சொல்லலாம்."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "இந்த நிகழ்வு கடுமையானதாகவோ உயிருக்கு ஆபத்தானதாகவோ இருந்தது "
        "என்று நீங்கள் சொல்வீர்களா?"
    ),
    "Please seek immediate medical attention now.": (
        "தயவுசெய்து இப்போதே உடனடியாக மருத்துவ உதவியை நாடுங்கள்."
    ),
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "நீங்கள் ஏற்கனவே பின்தொடர்புக்கு சம்மதித்திருந்தால், கூடுதல் தகவல்களை "
        "மிகவும் வசதியான நேரத்தில் சேகரிக்க ஏற்பாடு செய்வோம்."
    ),
    "Could you tell me when the event occurred?": "நிகழ்வு எப்போது நடந்தது என்று சொல்ல முடியுமா?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "நிகழ்வை ஆவணப்படுத்த உதவும் வேறு ஏதேனும் பகிர விரும்புகிறீர்களா?"
    ),
    "Of course.": "நிச்சயமாக.",
    "Please tell me any additional details you'd like included.": (
        "நீங்கள் சேர்க்க விரும்பும் கூடுதல் விவரங்களைத் தயவுசெய்து சொல்லுங்கள்."
    ),
    "Thank you for sharing this information with us.": "இந்தத் தகவலை எங்களுடன் பகிர்ந்ததற்கு நன்றி.",
    "We've prepared the report for review.": "மறுஆய்வுக்காக அறிக்கையைத் தயார் செய்துள்ளோம்.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "நீங்கள் பின்தொடர்புக்கு சம்மதி தெரிவித்திருந்தால், தேவைப்பட்டால் "
        "தொடர்புடைய குழு உங்களைத் தொடர்பு கொள்ளலாம்."
    ),
    "Is that right?": "அது சரியா?",
    "Have I got that right?": "நான் சரியாகப் புரிந்துகொண்டேனா?",
    "Have I recorded those correctly?": "அவற்றை நான் சரியாகப் பதிவு செய்தேனா?",
}

_TAMIL_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "முகவரி {0} ஆகப் படிக்கப்பட்டது.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "முகவரி இன்னும் {0} ஆகவே உள்ளது.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "உங்கள் மின்னஞ்சல் முகவரியை {0} என புரிந்துகொண்டேன்.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "உங்கள் மின்னஞ்சல் முகவரியை {0} என கேட்டேன்.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "நான் {0} கேட்டேன்."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "உங்கள் பெயர் {0} என்று என்னிடம் பதிவாகி உள்ளது.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "உங்கள் விவரங்களை நான் இப்படிப் பதிவு செய்துள்ளேன்: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "சரியாகப் பதிவு செய்துள்ளேனா என்று உறுதிப்படுத்த, மீண்டும் வாசித்துக் காட்டுகிறேன்: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "எனக்கு {0}-இலக்க எண் கிடைத்தது, அதன் இறுதிப் பகுதி {1} ஆகும்.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "மன்னிக்கவும், எனக்குப் புரியவில்லை - தயவுசெய்து சரியான {0} சொல்லுங்கள்.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "{0} பற்றி கொஞ்சம் கூடுதலாகச் சொல்ல முடியுமா?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "உங்கள் பிறந்த தேதி ({0}) மற்றும் உங்கள் வயது ({1} ஆண்டுகள்) பொருந்தவில்லை.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "எதைப் பதிவு செய்ய வேண்டும் - உங்கள் பிறந்த தேதியா அல்லது உங்கள் வயதா?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0}-க்கு, தயவுசெய்து மருந்தின் தயாரிப்பாளரையோ உங்கள் உள்ளூர் "
        "எதிர்மறை நிகழ்வு தெரிவிக்கும் சேனலையோ அணுகவும்.",
    ),
)

# ---- Telugu (te-IN) -----------------------------------------------------------

_TELUGU_SENTENCES: dict[str, str] = {
    "Thanks for that.": "దీనికి ధన్యవాదాలు.",
    "Got it, thank you.": "అర్థమైంది, ధన్యవాదాలు.",
    "Okay, thank you.": "సరే, ధన్యవాదాలు.",
    "Thanks for the details.": "వివరాలు చెప్పినందుకు ధన్యవాదాలు.",
    "Thanks for telling me, and I'm sorry you're going through this.": (
        "నాకు చెప్పినందుకు ధన్యవాదాలు; మీరు ఇది అనుభవిస్తుండటం నాకు బాధగా ఉంది."
    ),
    "I'll take a few details so we can record it safely.": (
        "దీన్ని సురక్షితంగా నమోదు చేయడానికి నేను కొన్ని వివరాలు తీసుకుంటాను."
    ),
    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects.": (
        "ముఖ్యమైనది: తెలిపిన దుష్ప్రభావాల మూల్యాంకనం మరియు చికిత్స కోసం దయచేసి "
        "అర్హమైన వైద్యుడిని లేదా ఆరోగ్య నిపుణుడిని సంప్రదించండి."
    ),
    "Do not self-diagnose or change your medication without medical advice.": (
        "వైద్య సలహా లేకుండా మీరే రోగనిర్ధారణ చేసుకోవద్దు, మీ మందును మార్చుకోవద్దు."
    ),
    "Which medicine or product do you think may have caused this?": (
        "ఏ మందు లేదా ఉత్పత్తి వల్ల ఇది జరిగి ఉండవచ్చునని మీరు అనుకుంటున్నారు?"
    ),
    "Can you tell me what side effect or reaction you noticed?": (
        "మీరు గమనించిన దుష్ప్రభావం లేదా ప్రతిచర్య ఏమిటో చెప్పగలరా?"
    ),
    "Are you the person who experienced this, or are you reporting it for someone else?": (
        "ఇది అనుభవించినది మీరేనా, లేక మరొకరి తరపున చెబుతున్నారా?"
    ),
    "What is the patient's full name?": "రోగి పూర్తి పేరు ఏమిటి?",
    "May I have your first and last name?": "మీ మొదటి మరియు చివరి పేరు తెలుసుకోవచ్చా?",
    "How old is the patient, or what is their date of birth?": (
        "రోగి వయస్సు ఎంత, లేదా వారి పుట్టిన తేదీ ఏమిటి?"
    ),
    "How old are you, or what is your date of birth?": (
        "మీ వయస్సు ఎంత, లేదా మీ పుట్టిన తేదీ ఏమిటి?"
    ),
    "And what is your name?": "మీ పేరు ఏమిటి?",
    "If we need a little more information later, is it alright if we contact you?": (
        "తర్వాత మాకు కొంచెం సమాచారం కావాలంటే, మిమ్మల్ని సంప్రదించవచ్చా?"
    ),
    "Would you prefer to be contacted by phone or by email?": (
        "ఫోన్ ద్వారానా, ఇమెయిల్ ద్వారానా సంప్రదించాలనుకుంటున్నారు?"
    ),
    "What phone number would you like us to use?": "మేము ఏ ఫోన్ నంబర్‌ను వాడాలి?",
    "What email address would you like us to use?": "మేము ఏ ఇమెయిల్ చిరునామాను వాడాలి?",
    "When did the side effect first start?": "దుష్ప్రభావం మొదటిసారి ఎప్పుడు ప్రారంభమైంది?",
    "Would you consider this problem serious?": "ఈ సమస్యను మీరు తీవ్రమైనదిగా భావిస్తున్నారా?",
    "How is the patient feeling now - have the symptoms improved, stayed the same, or gotten worse?": (
        "రోగి ఇప్పుడు ఎలా ఉన్నారు - లక్షణాలు మెరుగయ్యాయా, అలాగే ఉన్నాయా, లేక మరింత దిగజారాయా?"
    ),
    "How are you feeling now - have your symptoms improved, stayed the same, or gotten worse?": (
        "మీరు ఇప్పుడు ఎలా ఉన్నారు - మీ లక్షణాలు మెరుగయ్యాయా, అలాగే ఉన్నాయా, లేక మరింత దిగజారాయా?"
    ),
    "Sorry, I didn't catch that - please say yes if that's correct, or no if you'd like me to change it.": (
        "క్షమించండి, నాకు అర్థం కాలేదు - అది సరైనదైతే అవును అని, మార్చాలంటే కాదు అని చెప్పండి."
    ),
    "Is that correct?": "అది సరైనదేనా?",
    "Thank you - I've recorded everything we need for now.": (
        "ధన్యవాదాలు - ఇప్పటికి కావలసినవన్నీ నేను నమోదు చేసుకున్నాను."
    ),
    "A member of our team will follow up if needed.": (
        "అవసరమైతే మా బృందంలో ఒక సభ్యులు మిమ్మల్ని సంప్రదిస్తారు."
    ),
    "Take care.": "మీ జాగ్రత్తగా ఉండండి.",
    "Thank you.": "ధన్యవాదాలు.",
    "Before we finish, is there anything else you'd like to share about what happened?": (
        "ముగించే ముందు, ఏమి జరిగిందో దాని గురించి మరేదైనా పంచుకోవాలనుకుంటున్నారా?"
    ),
    "Thank you for sharing that.": "దానిని పంచుకున్నందుకు ధన్యవాదాలు.",
    "If this is a medical emergency, please seek immediate medical attention or call your local emergency services.": (
        "ఇది వైద్య అత్యవసర పరిస్థితి అయితే, దయచేసి వెంటనే వైద్య సహాయం పొందండి "
        "లేదా మీ స్థానిక అత్యవసర సేవలకు కాల్ చేయండి."
    ),
    "This appears potentially urgent.": "ఇది సంభావ్య అత్యవసర పరిస్థితిగా కనిపిస్తోంది.",
    "A healthcare professional should be contacted immediately.": (
        "వెంటనే ఒక ఆరోగ్య నిపుణుడిని సంప్రదించాలి."
    ),
    "I cannot provide personalized dosing, treatment recommendations, or medication changes.": (
        "వ్యక్తిగత మోతాదు, చికిత్స సిఫారసులు లేదా మందుల మార్పులను నేను సూచించలేను."
    ),
    "Please consult with a qualified healthcare professional for any changes to your medication regimen.": (
        "మీ మందుల విధానంలో ఏదైనా మార్పు కోసం దయచేసి అర్హమైన ఆరోగ్య "
        "నిపుణుడిని సంప్రదించండి."
    ),
    "I'm not sure how to interpret that.": "దానిని ఎలా అర్థం చేసుకోవాలో నాకు తోచలేదు.",
    "Could you please clarify?": "దయచేసి మీరు స్పష్టం చేయగలరా?",
    "Sorry, I didn't catch the product name.": "క్షమించండి, ఉత్పత్తి పేరు నాకు అర్థం కాలేదు.",
    "Which medicine were you using?": "మీరు ఏ మందు వాడుతున్నారు?",
    "Sorry, could you describe the side effect in a few words?": (
        "క్షమించండి, దుష్ప్రభావాన్ని కొన్ని మాటల్లో వివరించగలరా?"
    ),
    "Just to check - are you reporting your own experience, or someone else's?": (
        "నిశ్చయత కోసం - మీరు మీ సొంత అనుభవమా చెబుతున్నారు, లేక మరొకరిదా?"
    ),
    "Could you say your first and last name again?": (
        "మీ మొదటి మరియు చివరి పేరు మళ్లీ చెప్పగలరా?"
    ),
    "How old are you, in years?": "మీ వయస్సు ఎంత, సంవత్సరాల్లో?",
    "Could you tell me your age in years, or your date of birth?": (
        "సంవత్సరాల్లో మీ వయస్సు లేదా మీ పుట్టిన తేదీ చెప్పగలరా?"
    ),
    "Could you say the date of birth again?": "పుట్టిన తేదీ మళ్లీ చెప్పగలరా?",
    "Would it be alright if we contact you later - yes or no?": (
        "మేము తర్వాత మిమ్మల్ని సంప్రదిస్తే సరిపోతుందా - అవునా కాదా?"
    ),
    "Which is easier for you - phone, email, or both?": (
        "మీకు ఏది సులభం - ఫోన్, ఇమెయిల్, లేక రెండూ?"
    ),
    "Could you repeat the phone number, digit by digit?": (
        "ఫోన్ నంబర్‌ను అంకె అంకెగా మళ్లీ చెప్పగలరా?"
    ),
    "Could you repeat the email address?": "ఇమెయిల్ చిరునామా మళ్లీ చెప్పగలరా?",
    "About when did it start - for example today, yesterday, or a few days ago?": (
        "ఇది సుమారు ఎప్పుడు ప్రారంభమైంది - ఉదాహరణకు ఈరోజు, నిన్న, లేక కొన్ని రోజుల క్రితమా?"
    ),
    "Do you consider this serious - yes, no, or not sure?": (
        "దీన్ని మీరు తీవ్రమైనదిగా భావిస్తున్నారా - అవును, కాదు, లేక ఖచ్చితంగా తెలియదు?"
    ),
    "Are your symptoms better, about the same, or worse now?": (
        "మీ లక్షణాలు ఇప్పుడు మెరుగయ్యాయా, అలాగే ఉన్నాయా, లేక మరింత దిగజారాయా?"
    ),
    "Did you get any medical care for it - yes or no?": (
        "దానికి మీరు ఏదైనా వైద్య సంరక్షణ పొందారా - అవునా కాదా?"
    ),
    "How is the patient feeling now\u2014have the symptoms improved, stayed the same, or gotten worse?": (
        "రోగి ఇప్పుడు ఎలా ఉన్నారు - లక్షణాలు మెరుగయ్యాయా, అలాగే ఉన్నాయా, లేక మరింత దిగజారాయా?"
    ),
    "How are you feeling now\u2014have your symptoms improved, stayed the same, or gotten worse?": (
        "మీరు ఇప్పుడు ఎలా ఉన్నారు - మీ లక్షణాలు మెరుగయ్యాయా, అలాగే ఉన్నాయా, లేక మరింత దిగజారాయా?"
    ),
    "Would you prefer to be contacted by phone, by email, or both?": (
        "ఫోన్ ద్వారా, ఇమెయిల్ ద్వారా, లేక రెండింటి ద్వారా సంప్రదించాలనుకుంటున్నారు?"
    ),
    "Thank you for sharing this information.": "ఈ సమాచారాన్ని పంచుకున్నందుకు ధన్యవాదాలు.",
    "We've recorded what you've provided.": "మీరు అందించిన సమాచారాన్ని మేము నమోదు చేసుకున్నాము.",
    "Since you haven't consented to follow-up contact, we won't contact you using these details.": (
        "మీరు తదుపరి సంప్రదింపులకు అంగీకరించనందున, ఈ వివరాలను ఉపయోగించి "
        "మేము మిమ్మల్ని సంప్రదించము."
    ),
    "Thank you for your patience.": "మీ ఓపికికి ధన్యవాదాలు.",
    "We could not confirm some details, so your case will be reviewed by a member of our team.": (
        "కొన్ని వివరాలను మేము నిర్ధారించలేకపోయాము; కాబట్టి మీ కేసును మా "
        "బృందంలో ఒక సభ్యులు సమీక్షిస్తారు."
    ),
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow).
    "If that is the complete address, say yes - if more letters belong to it, please spell the address again.": (
        "అది పూర్తి చిరునామా అయితే అవును అనండి - మరిన్ని అక్షరాలు ఉంటే దయచేసి చిరునామాను మళ్లీ చెప్పండి."
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow).
    "If that is not correct, please spell it out again, one letter or number at a time.": (
        "అది సరైనది కాకపోతే, దయచేసి దాన్ని మళ్లీ ఒక్కొక్క అక్షరంగా లేదా సంఖ్యగా చెప్పండి."
    ),
    # Field-specific review recovery (kept in sync with the
    # confirmation review message in the conversational workflow):
    # the unresolved field is named and the reporter is invited to
    # resume or correct it.
    "We could not confirm the name.": (
        "మాకు పేరును ధృవీకరించడం సాధ్యం కాలేదు."
    ),
    "We could not confirm the phone number.": (
        "మాకు ఫోన్ నంబర్‌ను ధృవీకరించడం సాధ్యం కాలేదు."
    ),
    "We could not confirm the contact number.": (
        "మాకు కాంటాక్ట్ నంబర్‌ను ధృవీకరించడం సాధ్యం కాలేదు."
    ),
    "You can say it again now, or our team will collect it during review.": (
        "మీరు దీన్ని ఇప్పుడే మళ్లీ చెప్పవచ్చు, లేదా సమీక్ష సమయంలో మా బృందం దీన్ని సేకరిస్తుంది."
    ),
    "This MedVaani prototype is configured for Lantus SoloStar only.": (
        "ఈ MedVaani ప్రోటోటైప్ Lantus SoloStar కోసం మాత్రమే సిద్ధం చేయబడింది."
    ),
    "This prototype cannot continue this report.": (
        "ఈ ప్రోటోటైప్ ఈ నివేదికను కొనసాగించలేకపోతుంది."
    ),
    "Thank you for taking the time to speak with us.": "మాతో మాట్లాడటానికి సమయం కేటయించినందుకు ధన్యవాదాలు.",
    "I'm here to help record the event for review.": (
        "సమీక్ష కోసం సంఘటనను నమోదు చేయడంలో సహాయపడటానికి నేను ఇక్కడ ఉన్నాను."
    ),
    "I'll ask you a few questions, and we'll prepare a report for review.": (
        "నేను మీకు కొన్ని ప్రశ్నలు అడుగుతాను; సమీక్ష కోసం ఒక నివేదికను "
        "మేము తయారు చేస్తాము."
    ),
    "I'm not a substitute for a healthcare professional, and I can't provide medical advice.": (
        "నేను ఆరోగ్య నిపుణుడికి ప్రత్యామ్నాయం కాను; వైద్య సలహా ఇవ్వలేను."
    ),
    "Please tell me what happened, in your own words.": "ఏమి జరిగిందో మీ సొంత మాటల్లో చెప్పండి.",
    "Take your time\u2014I'm listening.": "మీ సమయం తీసుకోండి\u2014నేను వింటున్నాను.",
    "To help us record your report correctly, could you tell me the name of the medication involved?": (
        "మీ నివేదికను సరిగ్గా నమోదు చేయడంలో సహాయం కోసం, సంబంధిత మందు "
        "పేరు చెప్పగలరా?"
    ),
    "I'm sorry to hear that you've experienced this.": "మీరు ఇది అనుభవించారని తెలిసి బాధగా ఉంది.",
    "Just to make sure I have the right medication, is it Lantus SoloStar?": (
        "సరైన మందు నా దగ్గర ఉందేనా అని నిశ్చయించుకోవడానికి - అది Lantus SoloStarనా?"
    ),
    "May I ask whether you're the person who experienced this event, or whether you're reporting it on someone else's behalf?": (
        "ఈ సంఘటన అనుభవించినది మీరేనా, లేక మరొకరి తరపున చెబుతున్నారా అని అడగవచ్చా?"
    ),
    "Could you please tell me your full name?": "దయచేసి మీ పూర్తి పేరు చెప్పగలరా?",
    "Could you please share your contact number?": "దయచేసి మీ సంప్రదింపు నంబర్ చెప్పగలరా?",
    "Could you please tell me the patient's first and last name, along with their date of birth?": (
        "దయచేసి రోగి మొదటి మరియు చివరి పేరు, పుట్టిన తేదీతో సహా చెప్పగలరా?"
    ),
    "Could you please tell me your first and last name, along with your date of birth?": (
        "దయచేసి మీ మొదటి మరియు చివరి పేరు, పుట్టిన తేదీతో సహా చెప్పగలరా?"
    ),
    "Could you please tell me the patient's date of birth?": "దయచేసి రోగి పుట్టిన తేదీ చెప్పగలరా?",
    "Could you please tell me your date of birth?": "దయచేసి మీ పుట్టిన తేదీ చెప్పగలరా?",
    "Was it the name or the date of birth that was incorrect?": "పేరు తప్పా లేక పుట్టిన తేదీనా?",
    "Sorry, please tell me the name or the date of birth that was incorrect.": (
        "క్షమించండి, పేరు లేక పుట్టిన తేదీలో ఏది తప్పైందో చెప్పండి."
    ),
    "Just to check - was that another symptom, or a correction to what you described?": (
        "నిశ్చయత కోసం - అది మరొక లక్షణమా, లేక మీరు వివరించిన దానికి సవరణా?"
    ),
    "Sorry, please tell me a full date of birth.": "క్షమించండి, దయచేసి పూర్తి పుట్టిన తేదీ చెప్పండి.",
    "Sorry, I couldn't catch that.": (
        "క్షమించండి, నాకు అర్థం కాలేదు."
    ),
    "I could not hear the month clearly - please say the date of birth as day, month, and year.": (
        "నెల స్పష్టంగా వినబడలేదు - దయచేసి పుట్టిన తేదీని రోజు, నెల, సంవత్సరంగా చెప్పండి."
    ),
    "Could you please repeat it?": (
        "దయచేసి మళ్లీ చెప్పగలరా?"
    ),
    "I couldn't catch the email address.": (
        "ఇమెయిల్ చిరునామా నాకు అర్థం కాలేదు."
    ),
    "Could you please share the patient's contact number?": (
        "దయచేసి రోగి సంప్రదింపు నంబర్ చెప్పగలరా?"
    ),
    "Could you please share the best contact number for you?": (
        "దయచేసి మీకు అనుకూలమైన సంప్రదింపు నంబర్ చెప్పగలరా?"
    ),
    "Could you repeat the contact number, digit by digit?": (
        "సంప్రదింపు నంబర్‌ను అంకె అంకెగా మళ్లీ చెప్పగలరా?"
    ),
    "Would it be okay for the appropriate team to contact the patient about this report if any follow-up is needed?": (
        "మరిన్ని వివరాలు కావాలంటే, సంబంధిత బృందం ఈ నివేదిక గురించి రోగిని "
        "సంప్రదించవచ్చా?"
    ),
    "May I have the patient's email address, if available?": (
        "అందుబాటులో ఉంటే, రోగి ఇమెయిల్ చిరునామా తెలుసుకోవచ్చా?"
    ),
    "Please tell me the email address you'd like us to record.": (
        "మీరు నమోదు చేయించాలనుకునే ఇమెయిల్ చిరునామాను దయచేసి చెప్పండి."
    ),
    "Could you please spell out your email address, one letter or number at a time, with a short pause between letters?": (
        "మీ ఇమెయిల్ చిరునామాను ఒక్కొక్క అక్షరంగా లేదా అంకెగా చెప్పగలరా?"
    ),
    "You can say 'at the rate' for @ and 'dot' for a period.": (
        "'@' కోసం 'at the rate', పాయింట్ కోసం 'dot' అని చెప్పవచ్చు."
    ),
    "Would you say that the event was severe or life-threatening?": (
        "ఈ సంఘటన తీవ్రమైనదా లేక ప్రాణాంతకమైనదా అని మీరు అంటారా?"
    ),
    "Please seek immediate medical attention now.": "దయచేసి ఇప్పుడే వెంటనే వైద్య సహాయం పొందండి.",
    "If you've already consented to follow-up contact, we'll arrange to collect any additional information at a more convenient time.": (
        "మీరు ఇప్పటికే తదుపరి సంప్రదింపులకు అంగీకరించి ఉంటే, అదనపు సమాచారాన్ని "
        "మరింత అనుకూలమైన సమయంలో సేకరించేలా మేము ఏర్పాటు చేస్తాము."
    ),
    "Could you tell me when the event occurred?": "సంఘటన ఎప్పుడు జరిగిందో చెప్పగలరా?",
    "Is there anything else you'd like to share that may help us document the event?": (
        "సంఘటనను నమోదు చేయడంలో సహాయపడే మరేదైనా పంచుకోవాలనుకుంటున్నారా?"
    ),
    "Of course.": "ఖచ్చితంగా.",
    "Please tell me any additional details you'd like included.": (
        "మీరు చేర్చాలనుకునే అదనపు వివరాలను దయచేసి చెప్పండి."
    ),
    "Thank you for sharing this information with us.": "ఈ సమాచారాన్ని మాతో పంచుకున్నందుకు ధన్యవాదాలు.",
    "We've prepared the report for review.": "సమీక్ష కోసం నివేదికను మేము తయారు చేసాము.",
    "If you provided consent for follow-up contact, the appropriate team may contact you if needed.": (
        "మీరు తదుపరి సంప్రదింపులకు అంగీకరించి ఉంటే, అవసరమైతే సంబంధిత "
        "బృందం మిమ్మల్ని సంప్రదించవచ్చు."
    ),
    "Is that right?": "అది సరైనదేనా?",
    "Have I got that right?": "నేను సరిగ్గా అర్థం చేసుకున్నానా?",
    "Have I recorded those correctly?": "వాటిని నేను సరిగ్గా నమోదు చేశానా?",
}

_TELUGU_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # Incomplete-spelling clarification (kept in sync with
    # _incomplete_email_clarification in the conversational
    # workflow): a truncated re-spelling needs an explicit
    # completeness confirmation before it can commit.
    (
        re.compile(r"^The address reads as (.+)\.$"),
        "చిరునామా {0} గా చదవబడింది.",
    ),
    # Unchanged-candidate clarification (kept in sync with
    # _unchanged_email_clarification in the conversational workflow):
    # a rejected address can never be confirmed unchanged.
    (
        re.compile(r"^The address still reads as (.+)\.$"),
        "చిరునామా ఇప్పటికీ {0} గానే ఉంది.",
    ),
    # Email read-backs must match BEFORE the generic
    # "^I heard (.+)\.$" pattern, whose template would interpolate
    # the untranslated English middle into a localized sentence.
    (
        re.compile(r"^I understood your email address as (.+)\.$"),
        "మీ ఇమెయిల్ చిరునామాను {0} గా అర్థం చేసుకున్నాను.",
    ),
    (
        re.compile(r"^I heard your email address as (.+)\.$"),
        "మీ ఇమెయిల్ చిరునామాను {0} గా విన్నాను.",
    ),
    (re.compile(r"^I heard (.+)\.$"), "నేను {0} విన్నాను."),
    (
        re.compile(r"^I have your name as (.+)\.$"),
        "మీ పేరు {0} గా నా దగ్గర నమోదైంది.",
    ),
    (
        re.compile(r"^I have your details as (.+)\.$"),
        "మీ వివరాలను నేను ఇలా నమోదు చేశాను: {0}.",
    ),
    (
        re.compile(
            r"^Let me read that back to make sure I've recorded it correctly: (.+)\.$"
        ),
        "సరిగ్గా నమోదు చేశానో లేదో నిశ్చయించుకోవడానికి మళ్లీ చదివి వినిపిస్తాను: {0}.",
    ),
    (
        re.compile(r"^I have a (\d+)-digit number ending in (.+)\.$"),
        "నాకు {0} అంకెల నంబర్ అందింది, దాని చివరి భాగం {1}.",
    ),
    (
        re.compile(r"^Sorry, I didn't catch that - please tell me the correct (.+)\.$"),
        "క్షమించండి, నాకు అర్థం కాలేదు - దయచేసి సరైన {0} చెప్పండి.",
    ),
    (
        re.compile(r"^Could you tell me a bit more about the (.+)\?$"),
        "{0} గురించి కొంచెం మరిన్ని చెప్పగలరా?",
    ),
    (
        re.compile(
            r"^Your date of birth \((.+?)\) and your age \((.+?) years\) don't match\.$"
        ),
        "మీ పుట్టిన తేదీ ({0}) మరియు మీ వయస్సు ({1} సంవత్సరాలు) సరిపోలలేదు.",
    ),
    (
        re.compile(r"^Which should I record - your date of birth or your age\?$"),
        "ఏది నమోదు చేయాలి - మీ పుట్టిన తేదీనా లేక మీ వయస్సునా?",
    ),
    (
        re.compile(
            r"^For (.+), please contact the medicine's manufacturer "
            r"or your local adverse-event reporting channel\.$"
        ),
        "{0} కోసం, దయచేసి మందు తయారీదారుడిని లేదా మీ స్థానిక ప్రతికూల "
        "సంఘటన నివేదిక ఛానెల్‌ను సంప్రదించండి.",
    ),
)

_CATALOGS: dict[str, tuple[dict[str, str], tuple[tuple[Pattern[str], str], ...]]] = {
    "hi-IN": (_HINDI_SENTENCES, _HINDI_PATTERNS),
    "bn-IN": (_BENGALI_SENTENCES, _BENGALI_PATTERNS),
    "gu-IN": (_GUJARATI_SENTENCES, _GUJARATI_PATTERNS),
    "kn-IN": (_KANNADA_SENTENCES, _KANNADA_PATTERNS),
    "ml-IN": (_MALAYALAM_SENTENCES, _MALAYALAM_PATTERNS),
    "mr-IN": (_MARATHI_SENTENCES, _MARATHI_PATTERNS),
    "pa-IN": (_PUNJABI_SENTENCES, _PUNJABI_PATTERNS),
    "ta-IN": (_TAMIL_SENTENCES, _TAMIL_PATTERNS),
    "te-IN": (_TELUGU_SENTENCES, _TELUGU_PATTERNS),
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
