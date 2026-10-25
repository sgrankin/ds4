#!/usr/bin/env nu
# Generate one ordinary SystemOne request per line. Expected top choices are
# separate from model inputs, so cached/uncached runs see identical requests.
# Example: nu prefix_requests.nu .workspaces/clef-prefix/inputs --model clef
def main [output: path, --model: string = 'clef', --repeats: int = 1] {
    mkdir $output

    # Keep the original scheduling input for matched long-request timings.
    let chart_state = 1..250 | each { 'A customer was billed twice and needs a refund.' } | str join (char nl)
    let chart_state = $chart_state + (char nl)
    let routing = {
        type: 'choice'
        instructions: 'Which team should handle the request?'
        criteria: {billing: 'Payments, charges and refunds', technical: 'Broken software'}
    }

    # A bounded support record with a current issue and distinct historical
    # paragraphs. Most text is stable while question schemas are edited.
    let history = [
        'Account history, recorded before the active issue: Cedar Design has twelve seats on the Studio plan. The account owner is Maya Chen; support may contact her through the existing account email. The team uses the desktop application on macOS and exports monthly reports as PDF.'
        'The last ordinary invoice was INV-7301, dated October 1, for USD 240. Its authorized payment cleared on October 2. Finance reconciled that invoice without a dispute. A saved card ending in 1842 is the normal payment method. No bank account details are present in this record.'
        'Earlier support correspondence concerned changing the billing contact. Maya supplied the new contact on October 14; support verified the change on October 15. That request is closed. Historical references to refunds in the help-center links describe policy and do not themselves report an unresolved refund.'
        'The customer workspace contains design projects and client reports. The most recent successful export in the historical log was October 23 at 16:40. A routine service check on October 24 at 09:00 reported normal operation. Those observations predate the active issue and do not establish the current application status.'
        'Support workflow: route the active issue to the team that can resolve it. A payment or refund discrepancy belongs to billing. A reproducible application failure belongs to technical support. Account access requests belong to account support. Historical closed tickets must not override the active issue.'
    ] | str join ((char nl) + (char nl))
    let billing = 'ACTIVE ISSUE, October 25 at 10:00: Invoice INV-7319 for USD 240 was charged twice. The customer authorized one purchase and disputes the duplicate charge. The duplicate has not been refunded. The customer asks support to reverse the extra USD 240.'
    let technical = 'ACTIVE ISSUE, October 25 at 10:00: The desktop application now crashes every time a project is opened. The customer cannot export the report. The current invoice was charged once, and there is no payment dispute or refund request. The customer asks support to fix the application.'
    let billing_state = $billing + (char nl) + (char nl) + $history
    let technical_state = $technical + (char nl) + (char nl) + $history
    let chart = {model: $model, state: $chart_state, questions: {team: $routing}}
    let cases = [
        {name: 'chart-routing', request: $chart, expected_top: {team: 'billing'}}
        {
            name: 'chart-schema-grow'
            request: {
                model: $model
                state: $chart_state
                questions: {
                    owner: {
                        type: 'choice'
                        instructions: 'Select the team responsible for resolving the reported problem.'
                        criteria: {engineering: 'Broken software', billing: 'Payments, duplicate charges and refunds', account_access: 'Passwords and sign-in access'}
                    }
                    duplicate: {type: 'noul', instructions: 'Does the customer report being charged twice for one purchase?'}
                    discrepancy: {
                        type: 'score'
                        instructions: 'Which description fits the reported payment problem?'
                        criteria: ['No disputed charge', 'One duplicate charge', 'Several distinct duplicate charges']
                    }
                }
            }
            expected_top: {owner: 'billing', duplicate: 'true', discrepancy: '1'}
        }
        {
            name: 'narrative-routing'
            request: {model: $model, state: $billing_state, questions: {team: $routing}}
            expected_top: {team: 'billing'}
        }
        {
            name: 'narrative-schema-grow'
            request: {
                model: $model
                state: $billing_state
                questions: {
                    unresolved_refund: {type: 'noul', instructions: 'Does the active issue contain an unresolved request to refund a duplicate payment?'}
                    responsible_team: {
                        type: 'choice'
                        instructions: 'Which team should resolve the active issue?'
                        criteria: {account: 'Account access', technical: 'Application failures', billing: 'Payments or refunds'}
                    }
                }
            }
            expected_top: {unresolved_refund: 'true', responsible_team: 'billing'}
        }
        {
            name: 'early-fact-change'
            request: {model: $model, state: $technical_state, questions: {team: $routing}}
            expected_top: {team: 'technical'}
        }
        {
            name: 'shorter-ordinary'
            request: {
                model: $model
                state: 'The application crashes on launch. My invoice is correct. Please fix the crash.'
                questions: {team: $routing}
            }
            expected_top: {team: 'technical'}
        }
        {
            name: 'narrative-return-schema-shrink'
            request: {
                model: $model
                state: $technical_state
                questions: {technical_issue: {type: 'noul', instructions: 'Does the active issue report an application failure rather than a payment dispute?'}}
            }
            expected_top: {technical_issue: 'true'}
        }
        {
            name: 'billing-state-replacement'
            request: {
                model: $model
                state: $billing_state
                questions: {duplicate: {type: 'noul', instructions: 'Does the active issue report a duplicate payment?'}}
            }
            expected_top: {duplicate: 'true'}
        }
        {name: 'chart-return-identical', request: $chart, expected_top: {team: 'billing'}}
    ]
    let cases = 1..$repeats | each { $cases } | flatten
    $cases | get request | each { to json --raw } | str join (char nl) | save -f ($output | path join 'requests.jsonl')
    $cases | enumerate | each {|row|
        {index: $row.index, name: $row.item.name, expected_top: $row.item.expected_top}
    } | to json | save -f ($output | path join 'expected.json')
    print $'Wrote ($cases | length) requests and separate expected answers to ($output).'
}
