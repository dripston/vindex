"""
150 hand-authored prompts: 50 questions x 3 input forms.

  devanagari : Hindi in native Devanagari script
  roman      : the same question in Romanized Hindi -- no English words
  codemix    : natural Hinglish -- English and Romanized Hindi mixed the
               way people actually type it (not a translation exercise;
               written to sound like a real message)

None of this is machine-translated. Each of the 50 questions was authored
once per form, independently, in the register that form actually gets
used in (formal Hindi for devanagari; typing-on-a-phone Hindi for roman;
mixed everyday speech for codemix).

Categories (roughly 12-13 each):
  factual      -- general knowledge, similar in spirit to the original
                  10-question set
  banking      -- account/transaction/payment phrasing
  support      -- customer-support style complaints/requests
  open_ended   -- opinion/advice/how-to, no single correct answer

This module is imported by scripts/script_adherence.py. No LLM involved.
"""

# Each entry: id, category, devanagari, roman, codemix
CASES = [
    # ---------------- factual (13) ----------------
    {
        "id": "capital_maharashtra",
        "category": "factual",
        "devanagari": "महाराष्ट्र की राजधानी क्या है?",
        "roman": "Maharashtra ki rajdhani kya hai?",
        "codemix": "Maharashtra ka capital city kaunsa hai?",
    },
    {
        "id": "planets_count",
        "category": "factual",
        "devanagari": "हमारे सौर मंडल में कितने ग्रह हैं?",
        "roman": "Hamare saur mandal mein kitne grah hain?",
        "codemix": "Solar system mein total kitne planets hote hain?",
    },
    {
        "id": "freedom_year",
        "category": "factual",
        "devanagari": "भारत को स्वतंत्रता किस वर्ष मिली थी?",
        "roman": "Bharat ko azadi kis saal mili thi?",
        "codemix": "India ko independence kis year mein mili thi?",
    },
    {
        "id": "water_formula",
        "category": "factual",
        "devanagari": "पानी का रासायनिक सूत्र क्या है?",
        "roman": "Pani ka rasayanik sutra kya hai?",
        "codemix": "Water ka chemical formula kya hota hai?",
    },
    {
        "id": "national_animal",
        "category": "factual",
        "devanagari": "भारत का राष्ट्रीय पशु कौन सा है?",
        "roman": "Bharat ka rashtriya pashu kaun sa hai?",
        "codemix": "India ka national animal kya hai?",
    },
    {
        "id": "largest_ocean",
        "category": "factual",
        "devanagari": "पृथ्वी का सबसे बड़ा महासागर कौन सा है?",
        "roman": "Prithvi ka sabse bada mahasagar kaun sa hai?",
        "codemix": "World ka sabse bada ocean kaunsa hai?",
    },
    {
        "id": "sun_direction",
        "category": "factual",
        "devanagari": "सूर्य किस दिशा में उगता है?",
        "roman": "Suraj kis disha mein ugta hai?",
        "codemix": "Sun kis direction se rise hota hai?",
    },
    {
        "id": "leap_year_days",
        "category": "factual",
        "devanagari": "एक लीप वर्ष में कितने दिन होते हैं?",
        "roman": "Ek leap varsh mein kitne din hote hain?",
        "codemix": "Leap year mein kitne days hote hain?",
    },
    {
        "id": "father_of_nation",
        "category": "factual",
        "devanagari": "भारत में राष्ट्रपिता किसे कहा जाता है?",
        "roman": "Bharat mein rashtrapita kise kaha jata hai?",
        "codemix": "India ke Father of the Nation kaun the?",
    },
    {
        "id": "boiling_point",
        "category": "factual",
        "devanagari": "समुद्र तल पर पानी किस तापमान पर उबलता है?",
        "roman": "Samudra tal par pani kis taapman par ubalta hai?",
        "codemix": "Sea level par water kis temperature pe boil karta hai?",
    },
    {
        "id": "longest_river",
        "category": "factual",
        "devanagari": "भारत की सबसे लंबी नदी कौन सी है?",
        "roman": "Bharat ki sabse lambi nadi kaun si hai?",
        "codemix": "India ki longest river kaunsi hai?",
    },
    {
        "id": "tallest_mountain",
        "category": "factual",
        "devanagari": "दुनिया का सबसे ऊंचा पर्वत कौन सा है?",
        "roman": "Duniya ka sabse ooncha parvat kaun sa hai?",
        "codemix": "World ka highest mountain kaunsa hai?",
    },
    {
        "id": "human_bones",
        "category": "factual",
        "devanagari": "मानव शरीर में कितनी हड्डियां होती हैं?",
        "roman": "Manav sharir mein kitni haddiyan hoti hain?",
        "codemix": "Human body mein total kitni bones hoti hain?",
    },

    # ---------------- banking / transactional (13) ----------------
    {
        "id": "bank_balance_check",
        "category": "banking",
        "devanagari": "मैं अपना बैंक बैलेंस कैसे चेक करूं?",
        "roman": "Main apna bank balance kaise check karun?",
        "codemix": "Mera bank balance check karne ka sabse easy tarika kya hai?",
    },
    {
        "id": "upi_transfer_failed",
        "category": "banking",
        "devanagari": "मेरा यूपीआई ट्रांसफर फेल हो गया, पैसे कब वापस आएंगे?",
        "roman": "Mera UPI transfer fail ho gaya, paise kab wapas aayenge?",
        "codemix": "Sir mera UPI payment fail ho gaya but paise kat gaye, refund kab tak aayega?",
    },
    {
        "id": "credit_card_bill",
        "category": "banking",
        "devanagari": "क्रेडिट कार्ड का बिल समय पर न भरने पर क्या होता है?",
        "roman": "Credit card ka bill samay par na bharne par kya hota hai?",
        "codemix": "Agar credit card bill late pay karu toh kya penalty lagegi?",
    },
    {
        "id": "atm_pin_reset",
        "category": "banking",
        "devanagari": "मैं अपना एटीएम पिन कैसे रीसेट करूं?",
        "roman": "Main apna ATM pin kaise reset karun?",
        "codemix": "ATM pin bhool gaya hoon, reset kaise karna hai?",
    },
    {
        "id": "loan_emi_amount",
        "category": "banking",
        "devanagari": "पांच लाख के लोन पर मासिक किस्त कितनी बनेगी?",
        "roman": "Paanch lakh ke loan par mahina ki EMI kitni banegi?",
        "codemix": "5 lakh ka loan lu toh monthly EMI kitni aayegi approx?",
    },
    {
        "id": "fixed_deposit_interest",
        "category": "banking",
        "devanagari": "फिक्स्ड डिपॉजिट पर ब्याज दर कितनी है?",
        "roman": "Fixed deposit par byaj dar kitni hai?",
        "codemix": "FD karwane par interest rate kitna milta hai abhi?",
    },
    {
        "id": "cheque_bounce",
        "category": "banking",
        "devanagari": "अगर मेरा चेक बाउंस हो जाए तो क्या करना चाहिए?",
        "roman": "Agar mera cheque bounce ho jaye toh kya karna chahiye?",
        "codemix": "Mera cheque bounce ho gaya hai, ab kya steps lene padenge?",
    },
    {
        "id": "account_closure",
        "category": "banking",
        "devanagari": "मुझे अपना बैंक खाता बंद करना है, प्रक्रिया क्या है?",
        "roman": "Mujhe apna bank khata band karna hai, prakriya kya hai?",
        "codemix": "Mujhe apna savings account close karwana hai, process kya hoga?",
    },
    {
        "id": "neft_vs_rtgs",
        "category": "banking",
        "devanagari": "एनईएफटी और आरटीजीएस में क्या अंतर है?",
        "roman": "NEFT aur RTGS mein kya antar hai?",
        "codemix": "NEFT aur RTGS mein difference kya hota hai, konsa fast hai?",
    },
    {
        "id": "minimum_balance_penalty",
        "category": "banking",
        "devanagari": "न्यूनतम बैलेंस न रखने पर कितना जुर्माना लगता है?",
        "roman": "Nyuntam balance na rakhne par kitna jurmana lagta hai?",
        "codemix": "Minimum balance maintain nahi kiya toh kitna charge katega?",
    },
    {
        "id": "kyc_update",
        "category": "banking",
        "devanagari": "मुझे अपने खाते में केवाईसी अपडेट करनी है, कैसे करूं?",
        "roman": "Mujhe apne khate mein KYC update karni hai, kaise karun?",
        "codemix": "Mera KYC expire ho gaya hai, branch jaake update karna padega ya online ho jayega?",
    },
    {
        "id": "debit_card_block",
        "category": "banking",
        "devanagari": "अगर मेरा डेबिट कार्ड खो जाए तो मुझे तुरंत क्या करना चाहिए?",
        "roman": "Agar mera debit card kho jaye toh mujhe turant kya karna chahiye?",
        "codemix": "Mera debit card kho gaya hai, turant block kaise karu?",
    },
    {
        "id": "international_transfer",
        "category": "banking",
        "devanagari": "विदेश में पैसे भेजने के लिए कौन से दस्तावेज़ चाहिए?",
        "roman": "Videsh mein paise bhejne ke liye kaun se dastavez chahiye?",
        "codemix": "Foreign account mein money transfer karne ke liye kaunse documents chahiye?",
    },

    # ---------------- customer support (12) ----------------
    {
        "id": "order_not_delivered",
        "category": "support",
        "devanagari": "सर, मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ है, कृपया मदद करें।",
        "roman": "Sir, mera order abhi tak deliver nahi hua hai, kripya madad karein.",
        "codemix": "Sir mera order abhi tak deliver nahi hua, please help kare.",
    },
    {
        "id": "wrong_item_received",
        "category": "support",
        "devanagari": "मुझे गलत सामान मिला है, इसे कैसे बदलवाऊं?",
        "roman": "Mujhe galat samaan mila hai, ise kaise badalwaun?",
        "codemix": "Mujhe wrong item mila hai, return/exchange kaise karu?",
    },
    {
        "id": "refund_status",
        "category": "support",
        "devanagari": "मेरा रिफंड अभी तक नहीं आया, कृपया स्टेटस बताएं।",
        "roman": "Mera refund abhi tak nahi aaya, kripya status bataein.",
        "codemix": "Mera refund status kya hai, itna time ho gaya abhi tak credit nahi hua.",
    },
    {
        "id": "app_not_working",
        "category": "support",
        "devanagari": "आपकी ऐप बार-बार क्रैश हो रही है, कृपया ठीक करें।",
        "roman": "Aapki app baar-baar crash ho rahi hai, kripya theek karein.",
        "codemix": "Your app baar baar crash ho raha hai, please fix kariye jaldi.",
    },
    {
        "id": "cancel_subscription",
        "category": "support",
        "devanagari": "मुझे अपनी सदस्यता रद्द करनी है, कृपया प्रक्रिया बताएं।",
        "roman": "Mujhe apni sadasyata radd karni hai, kripya prakriya bataein.",
        "codemix": "Mujhe apna subscription cancel karna hai, process kya hai?",
    },
    {
        "id": "complaint_rude_staff",
        "category": "support",
        "devanagari": "आपके स्टाफ ने मेरे साथ बहुत बदतमीज़ी से बात की, मैं शिकायत दर्ज करना चाहता हूं।",
        "roman": "Aapke staff ne mere saath bahut badtameezi se baat ki, main shikayat darj karna chahta hoon.",
        "codemix": "Aapke staff ne bahut rude behave kiya mere saath, mujhe complaint file karni hai.",
    },
    {
        "id": "password_reset_not_working",
        "category": "support",
        "devanagari": "पासवर्ड रीसेट लिंक काम नहीं कर रहा है, क्या करूं?",
        "roman": "Password reset link kaam nahi kar raha hai, kya karun?",
        "codemix": "Password reset link kaam nahi kar raha, kya karu ab?",
    },
    {
        "id": "extra_charge_billed",
        "category": "support",
        "devanagari": "मेरे बिल में एक अतिरिक्त शुल्क जोड़ा गया है, यह क्यों?",
        "roman": "Mere bill mein ek atirikt shulk joda gaya hai, yeh kyun?",
        "codemix": "Mere bill mein extra charge add ho gaya hai, yeh kyu laga?",
    },
    {
        "id": "delivery_address_change",
        "category": "support",
        "devanagari": "क्या मैं अपने ऑर्डर का डिलीवरी पता बदल सकता हूं?",
        "roman": "Kya main apne order ka delivery pata badal sakta hoon?",
        "codemix": "Kya main apna delivery address change kar sakta hoon order ke baad?",
    },
    {
        "id": "warranty_claim",
        "category": "support",
        "devanagari": "मेरे उत्पाद की वारंटी अभी बाकी है, मैं दावा कैसे करूं?",
        "roman": "Mere utpad ki warranty abhi baaki hai, main dawa kaise karun?",
        "codemix": "Product ki warranty abhi valid hai, claim kaise karu?",
    },
    {
        "id": "otp_not_received",
        "category": "support",
        "devanagari": "मुझे ओटीपी नहीं मिल रहा, कृपया दोबारा भेजें।",
        "roman": "Mujhe OTP nahi mil raha, kripya dobara bhejein.",
        "codemix": "Mujhe OTP receive nahi ho raha, resend kar sakte ho please?",
    },
    {
        "id": "account_hacked",
        "category": "support",
        "devanagari": "मुझे लगता है मेरा खाता हैक हो गया है, तुरंत मदद चाहिए।",
        "roman": "Mujhe lagta hai mera khata hack ho gaya hai, turant madad chahiye.",
        "codemix": "Mera account hack ho gaya lagta hai, urgent help chahiye please.",
    },

    # ---------------- open-ended (12) ----------------
    {
        "id": "healthy_lifestyle_tips",
        "category": "open_ended",
        "devanagari": "स्वस्थ जीवनशैली अपनाने के लिए मुझे क्या करना चाहिए?",
        "roman": "Swasth jeevanshaili apnane ke liye mujhe kya karna chahiye?",
        "codemix": "Healthy lifestyle follow karne ke liye kya tips honge?",
    },
    {
        "id": "career_advice_engineering",
        "category": "open_ended",
        "devanagari": "इंजीनियरिंग के बाद करियर के कौन से विकल्प अच्छे हैं?",
        "roman": "Engineering ke baad career ke kaun se vikalp achhe hain?",
        "codemix": "Engineering complete karne ke baad best career options kya honge?",
    },
    {
        "id": "saving_money_tips",
        "category": "open_ended",
        "devanagari": "हर महीने पैसे बचाने के लिए अच्छे तरीके क्या हैं?",
        "roman": "Har mahine paise bachane ke liye achhe tarike kya hain?",
        "codemix": "Monthly savings badhane ke liye kya achhe tips hai?",
    },
    {
        "id": "learning_new_language",
        "category": "open_ended",
        "devanagari": "एक नई भाषा जल्दी सीखने का सबसे अच्छा तरीका क्या है?",
        "roman": "Ek nai bhasha jaldi seekhne ka sabse achha tarika kya hai?",
        "codemix": "New language fast seekhne ka best tarika kya hai?",
    },
    {
        "id": "starting_a_business",
        "category": "open_ended",
        "devanagari": "छोटा व्यवसाय शुरू करने के लिए मुझे किन बातों का ध्यान रखना चाहिए?",
        "roman": "Chhota vyavsay shuru karne ke liye mujhe kin baaton ka dhyan rakhna chahiye?",
        "codemix": "Small business start karne ke liye kin cheezon ka dhyan rakhna chahiye?",
    },
    {
        "id": "work_life_balance",
        "category": "open_ended",
        "devanagari": "काम और जीवन में संतुलन कैसे बनाए रखें?",
        "roman": "Kaam aur jeevan mein santulan kaise banaye rakhein?",
        "codemix": "Work life balance kaise maintain karein busy schedule mein?",
    },
    {
        "id": "improving_focus",
        "category": "open_ended",
        "devanagari": "पढ़ाई के दौरान ध्यान केंद्रित करने के उपाय क्या हैं?",
        "roman": "Padhai ke dauran dhyan kendrit karne ke upay kya hain?",
        "codemix": "Study karte time focus improve karne ke tips kya hai?",
    },
    {
        "id": "buying_first_car",
        "category": "open_ended",
        "devanagari": "पहली कार खरीदते समय किन बातों का ध्यान रखना चाहिए?",
        "roman": "Pehli car khareedte samay kin baaton ka dhyan rakhna chahiye?",
        "codemix": "First car buy karte time kin baaton ka khayal rakhna chahiye?",
    },
    {
        "id": "reducing_stress",
        "category": "open_ended",
        "devanagari": "रोज़मर्रा के तनाव को कम करने के अच्छे तरीके बताइए।",
        "roman": "Rozmarra ke tanav ko kam karne ke achhe tarike bataiye.",
        "codemix": "Daily stress kam karne ke liye kya achhe tarike hai?",
    },
    {
        "id": "choosing_college",
        "category": "open_ended",
        "devanagari": "सही कॉलेज चुनते समय किन बातों का ख्याल रखना चाहिए?",
        "roman": "Sahi college chunte samay kin baaton ka khyal rakhna chahiye?",
        "codemix": "Right college choose karte time kin factors ka khayal rakhna chahiye?",
    },
    {
        "id": "improving_english_speaking",
        "category": "open_ended",
        "devanagari": "अंग्रेज़ी बोलना बेहतर करने के लिए क्या अभ्यास करना चाहिए?",
        "roman": "Angrezi bolna behtar karne ke liye kya abhyas karna chahiye?",
        "codemix": "English speaking improve karne ke liye kaunsi practice karni chahiye?",
    },
    {
        "id": "planning_a_trip",
        "category": "open_ended",
        "devanagari": "कम बजट में अच्छी यात्रा की योजना कैसे बनाएं?",
        "roman": "Kam budget mein achhi yatra ki yojna kaise banayein?",
        "codemix": "Low budget mein achhi trip plan kaise karein?",
    },
]

INPUT_FORMS = ["devanagari", "roman", "codemix"]

assert len(CASES) == 50, f"expected 50 questions, got {len(CASES)}"


def build_prompts():
    """Returns 150 dicts: {case_id, category, input_form, text}."""
    prompts = []
    for c in CASES:
        for form in INPUT_FORMS:
            prompts.append({
                "case_id": c["id"],
                "category": c["category"],
                "input_form": form,
                "text": c[form],
            })
    return prompts


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    prompts = build_prompts()
    print(f"{len(CASES)} questions x {len(INPUT_FORMS)} forms = {len(prompts)} prompts")
    from collections import Counter
    print("by category:", Counter(c["category"] for c in CASES))
    print("by form:", Counter(p["input_form"] for p in prompts))
