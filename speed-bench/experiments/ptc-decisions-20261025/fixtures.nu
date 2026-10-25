#!/usr/bin/env nu
# Fixed synthetic inputs and an external oracle. Gold labels never enter a model.
def main [output: path] {
    mkdir $output
    let policy = 'An active account takeover is a real incident where an unauthorized person still controls the account or an unauthorized session remains active. Exclude incidents already secured or closed, authorized activity, hypothetical worries, simulations and training examples. A large charge alone does not establish takeover. Evaluate each ticket independently using its current note.'
    let rows = [
        [id active charge_cents note];
        ['Q-AX7' true 24900 'A stranger changed my recovery email this morning. I still cannot sign in, and the unknown device remains connected. I did not authorize the listed purchase. Support has not secured the account.']
        ['Q-MR2' false 990000 'Last week a stranger accessed this account, but support restored it, rotated the password and revoked every unknown session. The incident is closed. Today I only need a copy of that old invoice.']
        ['Q-HK9' false 4500 'What should I do if someone ever steals my account? Nothing has happened and no login or purchase is disputed. This is a hypothetical question about preparation.']
        ['Q-DV4' true 78900 'Unknown sessions keep appearing after I try to revoke them. Someone is currently sending messages from my account and making purchases. I have not regained control; the recovery request is still unresolved.']
        ['Q-ZP8' false 550000 'This is a fictional example for a staff security exercise: an attacker changes the recovery email and buys a laptop. No real customer account was accessed and no actual purchase took place.']
        ['Q-BN3' false 8200 'The application crashes whenever I export a report. I can sign in normally, all devices are mine and the recovery details are unchanged. Please investigate the software failure.']
        ['Q-LF6' true 119900 'The intruder still controls my account after changing both the password and phone number. The listed charge was not mine. Support confirmed unauthorized access today, but containment has not happened.']
        ['Q-TS5' false 123000 'I authorized one purchase but the payment processor billed it twice. There is no suspicious login, no changed recovery information and no loss of account control. Please refund the duplicate.']
        ['Q-EC1' false 4000 'A login alert worried me until security verified that it was our company VPN on my own laptop. The session and purchase were authorized. There is no security incident.']
        ['Q-WJ8' false 7600 'My partner made this purchase using the shared account with my permission. I recognize all sessions and retain control. I only want to change the shipping address.']
        ['Q-RC4' true 3400 'I am watching an unfamiliar session change settings right now. It is not a simulation and I did not grant access. My recovery attempts have failed and that session is still active.']
        ['Q-UG2' false 999000 'The compromised account was permanently closed last month and all access was terminated. No active account remains. This ticket asks for historical records for an insurance claim.']
        ['Q-PA9' false 21000 'I changed my own email and password while moving to a new phone. All changes were authorized and I can still access the account. Please help update my billing contact.']
        ['Q-NY3' false 6500 'The password-reset page is slow, but I can sign in through the existing session. No unfamiliar device or unauthorized action has been found. This is a technical support request.']
        ['Q-FS7' false 300000 'The customer was afraid that a leaked password might be used. Support reset it before any unknown login occurred. The account is now secured and there is no evidence of unauthorized access.']
        ['Q-KU6' true 66500 'An unauthorized person is using the account to contact customers today. The recovery email has been replaced and I cannot remove the unknown session. This remains an active unresolved takeover.']
        ['Q-CB5' false 11000 'Our help-center article describes how an attacker could change a recovery email. I am asking whether the article is up to date. My own account has no suspicious activity.']
        ['Q-XA2' false 9900 'A developer reproduced an account takeover inside an isolated test tenant using fabricated users. The production account is unaffected. This ticket requests documentation of the test.']
        ['Q-JM8' false 17000 'I cannot find a receipt for an authorized purchase. My usual email and phone are intact, I recognize the sessions and I have full access. Please resend the receipt.']
        ['Q-VT1' false 750000 'Security removed the attacker yesterday, restored my recovery details and confirmed that no unknown session remains. The account is safe now; only reimbursement paperwork remains open.']
        ['Q-GD4' true 42500 'There is an ongoing unauthorized login on a device I do not own. Purchases continue while I write this ticket. Nobody has restored control or ended the unknown session yet.']
        ['Q-SL9' false 2400 'An automatic timeout signed me out of the desktop app. Signing back in worked and every device is mine. There is no unauthorized access; I want a longer session timeout.']
        ['Q-HE3' false 910000 'This is a role-play transcript for a training class. The speaker says an intruder still controls the account, but the account and purchase are invented. Do not treat this as a customer incident.']
        ['Q-OR6' false 5200 'I reported an unfamiliar login, but later recognized it as my own tablet. I approved its purchase and recovery details were never altered. Please close the mistaken alert.']
    ]
    let history = 'Background recorded before the current note: the customer uses the standard workspace plan, receives monthly invoices and normally exports reports at the end of each week. Earlier routine service checks reported that exports, notifications and ordinary payment processing were available. The support record also contains an old request to update an invoice contact. That administrative request was completed before the events in the current note. These historical service observations do not resolve or change the current issue.'
    let tail = 'The original correspondence is retained in support storage. This background paragraph describes routine account administration only. The current note above determines whether there is an active incident; do not infer current security status from an old successful service check or from the size of the listed charge.'
    let schema = {
        type: choice,
        instructions: 'Apply the policy in the input to the ticket identified below. Does that ticket qualify as an active account takeover?',
        choices: [
            {value: active, description: 'A real unresolved takeover or currently active unauthorized session'},
            {value: other, description: 'Resolved, authorized, hypothetical, simulated, or another kind of support issue'}
        ]
    }
    let prompt = 'Prepare the active account-takeover handoff from the attached queue. Report the qualifying ticket IDs, their count and the largest reported charge in cents. Exclude resolved incidents, authorized activity, simulations and hypothetical concerns. Finish with two sentences for the security team; do not claim that any account has already been locked or refunded. Return a JSON object with ids, count, max_charge_cents and handoff.'
    let cases = [
        {name: ordinary, rows: ($rows | first 4), long: false},
        {name: favorable, rows: $rows, long: true}
    ] | each {|case|
        let records = $case.rows | each {|r|
            {
                id: $r.id,
                charge_cents: $r.charge_cents,
                note: (if $case.long { $r.note + "\n\n" + $history + "\n\n" + $tail } else { $r.note })
            }
        }
        let active = $case.rows | where active
        {
            name: $case.name,
            queue: {policy: $policy, records: $records},
            manifest: {policy: $policy, records: ($records | select id charge_cents)},
            schema: $schema,
            prompt: $prompt,
            expected: {
                ids: ($active | get id | sort),
                count: ($active | length),
                max_charge_cents: ($active | get charge_cents | math max)
            }
        }
    }
    $cases | to json --indent 2 | save --force ($output | path join 'fixtures.json')
    print ($cases | each {|c| {case: $c.name, records: ($c.queue.records | length), active: $c.expected.count}})
}
