#!/usr/bin/env nu
# Real-model HTTP qualification. Run against a server with Clef-Flash loaded.
use std/assert

def post [url: string, endpoint: string, request: record] {
    http post --allow-errors --full --content-type application/json $"($url)($endpoint)" $request
}

def main [--url: string = 'http://127.0.0.1:19285', --model: string = 'clef-flash'] {
    let models = http get $"($url)/v1/models"
    let loaded = $models.data | where id == $model | first
    assert ('decisions' in $loaded.capabilities)
    let listed = http get $"($url)/v1/models/($model)"
    assert equal $listed.id $model

    let request = {
        model: $model,
        state: 'The customer was charged twice and needs the duplicate payment refunded.',
        questions: {
            department: {
                type: choice,
                instructions: 'Which department should handle this request?',
                criteria: {technical: 'Software malfunction', billing: 'Payments and refunds', account: 'Login problems'}
            },
            paid: {type: noul, instructions: 'Was the customer charged money?'}
        }
    }
    let system = post $url /v1/systemone $request
    assert equal $system.status 200
    assert equal $system.body.answers.department.choice billing
    assert ($system.body.answers.paid.noul > 0.5)
    assert equal $system.body.usage.output_tokens 0
    assert ($system.body.usage.input_tokens > 0)
    let sum = $system.body.answers.department.probabilities | values | math sum
    assert (($sum - 1 | math abs) < 0.00001)

    let openai = post $url /v1/decisions {
        model: $model,
        input: 'The customer was charged twice and needs the duplicate payment refunded.',
        questions: [
            {type: predicate, name: repeated, instructions: 'Was the customer charged money?'},
            {type: choice, name: repeated, instructions: 'Choose the correct department.', choices: [
                {value: billing, description: 'Payments and refunds'},
                {value: account, description: 'Login problems'}
            ]},
            {type: score, instructions: 'Rate the impact.', levels: [
                {label: minor, description: 'No money charged'},
                {label: material, description: 'A duplicate payment needs a refund'}
            ]},
            {type: choice, instructions: 'Select the proposition that applies.', choices: [
                {value: true, description: 'The customer was charged'},
                {value: 'true', description: 'The customer was not charged'}
            ]}
        ]
    }
    assert equal $openai.status 200
    assert equal ($openai.body.answers | length) 4
    assert equal ($openai.body.answers | get type) [predicate choice score choice]
    assert equal ($openai.body.answers | get name) [repeated repeated null null]
    assert equal $openai.body.answers.1.choice billing
    assert equal $openai.body.answers.3.probabilities.0.value true
    assert equal $openai.body.answers.3.probabilities.1.value 'true'
    assert equal $openai.body.usage.input_tokens_details.cached_tokens 0

    let unknown = post $url /v1/systemone ($request | update model missing-model)
    assert equal $unknown.status 404
    let box = post $url /v1/systemone {model: $model, state: text, questions: {location: {type: box, instructions: 'Find it'}}}
    assert equal $box.status 400
    let oversized = post $url /v1/systemone ($request | update state (1..20000 | each {'hello'} | str join ' '))
    assert equal $oversized.status 422
    print 'Decision HTTP checks pass: discovery, both APIs, task answers, ordered/typed values and explicit model/type/context errors.'
}
