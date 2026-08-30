import '../models/screening_models.dart';

const List<String> aq10Options = [
  'Definitely Agree',
  'Slightly Agree',
  'Slightly Disagree',
  'Definitely Disagree',
];

const List<ScreeningQuestion> qchat10Questions = [
  ScreeningQuestion(
    id: 'qchat10_q1',
    text: 'Does your child look at you when you call his/her name?',
    options: ['Always', 'Usually', 'Sometimes', 'Rarely', 'Never'],
  ),
  ScreeningQuestion(
    id: 'qchat10_q2',
    text: 'How easy is it for you to get eye contact with your child?',
    options: [
      'Very easy',
      'Quite easy',
      'Quite difficult',
      'Very difficult',
      'Impossible',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q3',
    text: 'Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q4',
    text: 'Does your child point to share interest with you? (e.g. pointing at an interesting sight)',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q5',
    text: 'Does your child pretend? (e.g. care for dolls, talk on a toy phone)',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q6',
    text: 'Does your child follow where you’re looking?',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q7',
    text: 'If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)',
    options: ['Always', 'Usually', 'Sometimes', 'Rarely', 'Never'],
  ),
  ScreeningQuestion(
    id: 'qchat10_q8',
    text: 'Would you describe your child’s first words as:',
    options: [
      'Very typical',
      'Quite typical',
      'Slightly unusual',
      'Very unusual',
      'My child doesn’t speak',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q9',
    text: 'Does your child use simple gestures? (e.g. wave goodbye)',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
  ScreeningQuestion(
    id: 'qchat10_q10',
    text: 'Does your child stare at nothing with no apparent purpose?',
    options: [
      'Many times a day',
      'A few times a day',
      'A few times a week',
      'Less than once a week',
      'Never',
    ],
  ),
];

const List<ScreeningQuestion> aq10ChildQuestions = [
  ScreeningQuestion(
    id: 'aq10_child_q1',
    text: 'S/he often notices small sounds when others do not',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q2',
    text: 'S/he usually concentrates more on the whole picture, rather than the small details',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q3',
    text: 'In a social group, s/he can easily keep track of several different people’s conversations',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q4',
    text:
        'S/he finds it easy to go back and forth between different activities',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q5',
    text:
        'S/he doesn’t know how to keep a conversation going with his/her peers',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q6',
    text: 'S/he is good at social chit-chat',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q7',
    text: 'When s/he is read a story, s/he finds it difficult to work out the character’s intentions or feelings',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q8',
    text: 'When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q9',
    text: 'S/he finds it easy to work out what someone is thinking or feeling just by looking at their face',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_child_q10',
    text: 'S/he finds it hard to make new friends',
    options: aq10Options,
  ),
];

const List<ScreeningQuestion> aq10AdolescentQuestions = [
  ScreeningQuestion(
    id: 'aq10_adolescent_q1',
    text: 'S/he notices patterns in things all the time',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q2',
    text: 'S/he usually concentrates more on the whole picture, rather than the small details',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q3',
    text: 'In a social group, s/he can easily keep track of several different people’s conversations',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q4',
    text: 'If there is an interruption, s/he can switch back to what s/he was doing very quickly',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q5',
    text: 'S/he frequently finds that s/he doesn’t know how to keep a conversation going',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q6',
    text: 'S/he is good at social chit-chat',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q7',
    text: 'When s/he was younger, s/he used to enjoy playing games involving pretending with other children',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q8',
    text: 'S/he finds it difficult to imagine what it would be like to be someone else',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q9',
    text: 'S/he finds social situations easy',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adolescent_q10',
    text: 'S/he finds it hard to make new friends',
    options: aq10Options,
  ),
];

const List<ScreeningQuestion> aq10AdultQuestions = [
  ScreeningQuestion(
    id: 'aq10_adult_q1',
    text: 'I often notice small sounds when others do not',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q2',
    text: 'I usually concentrate more on the whole picture, rather than the small details',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q3',
    text: 'I find it easy to do more than one thing at once',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q4',
    text: 'If there is an interruption, I can switch back to what I was doing very quickly',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q5',
    text: 'I find it easy to ‘read between the lines’ when someone is talking to me',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q6',
    text: 'I know how to tell if someone listening to me is getting bored',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q7',
    text: 'When I’m reading a story I find it difficult to work out the characters’ intentions',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q8',
    text: 'I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc)',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q9',
    text: 'I find it easy to work out what someone is thinking or feeling just by looking at their face',
    options: aq10Options,
  ),
  ScreeningQuestion(
    id: 'aq10_adult_q10',
    text: 'I find it difficult to work out people’s intentions',
    options: aq10Options,
  ),
];

const Map<QuestionnaireType, List<ScreeningQuestion>> questionBanks = {
  QuestionnaireType.qchat10: qchat10Questions,
  QuestionnaireType.aq10Child: aq10ChildQuestions,
  QuestionnaireType.aq10Adolescent: aq10AdolescentQuestions,
  QuestionnaireType.aq10Adult: aq10AdultQuestions,
};
